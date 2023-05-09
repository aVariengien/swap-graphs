# %%
import gc
import itertools
import os
import random
import random as rd
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union

import circuitsvis as cv
import datasets
import einops
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm.auto as tqdm
import transformer_lens
import transformer_lens.utils as utils
from attrs import define, field
from swap_graphs.datasets.ioi.ioi_utils import (
    logit_diff,
)
from jaxtyping import Float, Int

from swap_graphs.core import (
    NOT_A_HEAD,
    ActivationStore,
    CompMetric,
    ModelComponent,
    SwapGraph,
    SgraphDataset,
    TokDataset,
    WildPosition,
    break_long_str,
    compute_clustering_metrics,
    find_important_components,
)
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics.cluster import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    rand_score,
)
from torch.utils.data import DataLoader
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
)
from transformer_lens.hook_points import (  # Hooking utilities
    HookedRootModule,
    HookPoint,
)
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from umap import UMAP
from swap_graphs.utils import (
    KL_div_sim,
    L2_dist,
    L2_dist_in_context,
    compo_name_to_object,
    component_name_to_idx,
    create_random_communities,
    get_components_at_position,
    imshow,
    line,
    load_object,
    plotHistLogLog,
    print_gpu_mem,
    save_object,
    scatter,
    show_attn,
    show_mtx,
    clean_gpu_mem,
)

torch.set_grad_enabled(False)


def average_class_entropy(classes: Dict[ModelComponent, Dict[int, int]]):
    """We measure the entropy of a particular partition of a set S partitioned into the partition (C_i)i.
    Each C_i is permuted independantly and uniformly at random. We compute the entropy of the resulting permutation of S.
    e.g. if C_0=S, then the entropy is log(|S|!). If C_i are singletons, then the entropy is 0.
    """
    all_entropies = []
    for c in classes.values():
        class_entropy = 0
        reverse_d = {}
        for k, v in c.items():
            if v not in reverse_d:
                reverse_d[v] = 0
            reverse_d[v] += 1
        for size in reverse_d.values():
            for k in range(2, size + 1):
                class_entropy += np.log2(k)
        all_entropies.append(class_entropy)
    return np.mean(all_entropies)


def average_cluster_size(classes: Dict[ModelComponent, Dict[int, int]], n_samples: int):
    sizes = []
    for c in classes.values():
        reverse_d = {}
        for k, v in c.items():
            if v not in reverse_d:
                reverse_d[v] = 0
            reverse_d[v] += 1
        sizes += list(reverse_d.values())
    return np.mean(sizes) / n_samples


## Test the random communities and the entropy and cluster size computation


test_list_components = [
    ModelComponent(position=42, layer=l, name="z", head=h, position_label="test")
    for l in range(12)
    for h in range(12)
]

random_comus = create_random_communities(
    test_list_components, n_samples=100, n_classes=20
)
assert (
    average_cluster_size(random_comus, 100) >= 0.04
    and average_cluster_size(random_comus, 100) <= 0.06
)

random_comus = create_random_communities(
    test_list_components, n_samples=100, n_classes=5
)

assert (
    average_cluster_size(random_comus, 100) >= 0.18
    and average_cluster_size(random_comus, 100) <= 0.22
)

random_comus = create_random_communities(
    test_list_components, n_samples=100, n_classes=1
)  # all samples in the same class
assert (
    abs(average_class_entropy(random_comus) - 524.76499) < 1e-3
)  #  524.76499 = log(100!)

random_comus = create_random_communities(
    test_list_components, n_samples=100, n_classes=int(1e9)
)  # in the limit, no collision, one sample per classes, the partitions are made of singletons.

assert average_class_entropy(random_comus) < 1e-5

# %%


def get_dist_percentile(X, percentile=25):
    idx_i = np.random.choice(X.shape[0], size=1000, replace=True)
    idx_j = np.random.choice(X.shape[0], size=1000, replace=True)
    dist = np.linalg.norm(X[idx_i] - X[idx_j], axis=1)
    return np.percentile(dist, percentile)


def hierarchical_clustering(
    model: HookedTransformer,
    dataset: SgraphDataset,
    list_of_components: List[ModelComponent],
    progress_bar: bool = True,
    threshold_factor: float = 2.0,
    linkage: str = "ward",
) -> Dict[ModelComponent, Dict[int, int]]:
    """Compute the ward clustering of the activations of the components in list_of_components. Clustering is done using the L2 distance between the activations."""
    activation_store = ActivationStore(
        model=model, dataset=dataset.tok_dataset, listOfComponents=list_of_components
    )
    target_idx = [i for i in range(len(dataset))]

    clusterings = {}
    for c in tqdm.tqdm(list_of_components, disable=not progress_bar):
        seq_idx = c.position.positions_from_idx(target_idx)
        if c.is_head():
            activations = activation_store.transformerLensCache[
                c.hook_name
            ][  #  dim: (batch, seq, head, d_head or d_model). We extract the head at the right position
                range(len(dataset)), seq_idx, c.head, :
            ]
        else:
            activations = activation_store.transformerLensCache[
                c.hook_name
            ][  #  dim: (batch, seq, d_mlp or d_model).
                range(len(dataset)), seq_idx, :
            ]
        activations = activations.cpu().numpy()
        threshold = get_dist_percentile(activations, percentile=90)
        ward = AgglomerativeClustering(
            linkage=linkage,  # type: ignore
            distance_threshold=threshold_factor * threshold,
            n_clusters=None,
        )
        ward.fit(activations)
        clusterings[c] = {i: ward.labels_[i] for i in range(len(ward.labels_))}
    del activation_store
    clean_gpu_mem()
    return clusterings


def create_sgraph_communities(
    model,
    list_of_components: List[ModelComponent],
    dataset: SgraphDataset,
    all_sgraph_data: Dict,
    resolution=1.0,
):
    """Create the communities of the PatchingNetwork at a given resolution using the Luvain algorithm."""
    sgraph_communities = {}

    comp_metric = partial(
        KL_div_sim,
        position_to_evaluate=WildPosition(-42, label="DUMMY_POSITION"),
    )

    for c in list_of_components:
        sgraph = SwapGraph(
            model=model,
            tok_dataset=dataset.tok_dataset.cpu(),
            comp_metric=comp_metric,  # type: ignore
            patchedComponents=[c],
            proba_edge=1.0,
        )
        sgraph.load_comp_metric_edges(all_sgraph_data[f"{c}"]["sgraph_edges"])  # 0.06s
        sgraph.compute_weights()  # 0.08s
        commu_list = sgraph.compute_communities(resolution=resolution)

        sgraph_communities[c] = {i: commu_list[i] for i in range(len(commu_list))}

    return sgraph_communities
