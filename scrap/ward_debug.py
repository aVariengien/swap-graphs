# %%
import gc
import itertools
import os
import random
import random as rd
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple, Union

import circuitsvis as cv
import datasets
import einops
import fire
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm.auto as tqdm
import transformer_lens
import transformer_lens.utils as utils
from attrs import define, field
from swap_graphs.communities_utils import (
    average_class_entropy,
    average_cluster_size,
    create_sgraph_communities,
    hierarchical_clustering,
)
from swap_graphs.datasets.ioi.ioi_utils import (
    IOIDataset,
    logit_diff,
    probs,
    assert_model_perf_ioi,
)
from jaxtyping import Float, Int
from names_generator import generate_name
from swap_graphs.PatchedModel import PatchedModel
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
    L2_dist,  # type: ignore
    L2_dist_in_context,
    clean_gpu_mem,
    compo_name_to_object,
    component_name_to_idx,
    create_random_communities,
    get_components_at_position,
    imshow,
    line,
    load_object,
    plotHistLogLog,
    print_gpu_mem,
    print_time,
    save_object,
    scatter,
    show_attn,
    show_mtx,
    load_config,
)

torch.set_grad_enabled(False)


# %%


def sweep_scrub(
    model: HookedTransformer,
    sgraph_dataset: SgraphDataset,
    ioi_dataset: IOIDataset,
    classes: Dict[ModelComponent, Dict[int, int]],
    progress_bar=True,
):
    """Scrub by compunity until layer L. Do this for L=0 to nb_layers."""
    patched_model = PatchedModel(
        model=model, sgraph_dataset=sgraph_dataset, communities=classes
    )

    max_L = max([c.layer for c in classes.keys()])
    compos_list = list(classes.keys())

    results = {}
    results["logit_diff"] = []
    results["io_prob"] = []
    results["s_prob"] = []

    for L in tqdm.tqdm(
        range(max_L - 2, max_L), disable=not progress_bar
    ):  # TODO change start
        print(f"Scrubbing up to layer {L}")
        compos_up_to_L = [c for c in compos_list if c.layer <= L]
        model.reset_hooks()
        patched_model.model.reset_hooks()
        patched_model.scrub_by_communities(list_of_components=compos_up_to_L)

        ld = logit_diff(patched_model, ioi_dataset)
        io_prob = probs(patched_model, ioi_dataset, type="io")
        s_prob = probs(patched_model, ioi_dataset, type="s")
        # print(f"Logit diff: {ld.item()}, IO prob: {io_prob.item()} S prob: {s_prob.item()}")  # type: ignore
        results["logit_diff"].append(ld.item())  # type: ignore
        results["io_prob"].append(io_prob.item())  # type: ignore
        results["s_prob"].append(s_prob.item())  # type: ignore
    return results


# %%
xp_name = "gpt2-small-IOI-compassionate_einstein"  # "EleutherAI-gpt-neo-2.7B-IOI-serene_murdock" #gpt2-small-IOI-compassionate_einstein
model_name = "gpt2-small"
xp_path = "../xp"

path, model_name, MODEL_NAME, dataset_name = load_config(
    xp_name, xp_path, model_name  # type: ignore
)

sgraph_dataset = load_object(path, "sgraph_dataset.pkl")
ioi_dataset = load_object(path, "ioi_dataset.pkl")
comp_metric = load_object(path, "comp_metric.pkl")
all_sgraph_data = load_object(path, "all_data.pkl")

if hasattr(ioi_dataset, "prompts_toks"):  # for backward compatibility
    ioi_dataset.prompts_tok = ioi_dataset.prompts_toks


raw_model = HookedTransformer.from_pretrained(
    model_name, device="cpu"
)  # raw model to hard reset the hooks
print_gpu_mem("after loading raw model")
model = deepcopy(raw_model)
model.to("cuda")

end_position = WildPosition(position=ioi_dataset.word_idx["END"], label="END")

list_components = [
    compo_name_to_object(c, end_position, raw_model.cfg.n_heads)
    for c in all_sgraph_data.keys()
]
# %%


ward_threshold_factors = [
    3.7,
    0.6,
]


# %%


def print_clusters(clusters):
    activation_store = ActivationStore(
        model=model,
        dataset=sgraph_dataset.tok_dataset,
        listOfComponents=None,
        force_cache_all=True,
    )
    c = ModelComponent(
        position=end_position, position_label="END", layer=9, head=9, name="z"
    )

    target_idx = [i for i in range(len(sgraph_dataset))]
    seq_idx = c.position.positions_from_idx(target_idx)
    activations = activation_store.transformerLensCache[
        c.hook_name
    ][  #  dim: (batch, seq, d_mlp or d_model).
        range(len(sgraph_dataset)), seq_idx, c.head, :
    ]
    print(activations.shape)

    cluster = clusters[c]
    # plot the UMAP of the activations with the
    umap = UMAP(
        n_components=2, n_neighbors=10, min_dist=0.1, metric="cosine", random_state=42
    )
    umap_activations = umap.fit_transform(activations.cpu().numpy())
    fig = px.scatter(
        x=umap_activations[:, 0],
        y=umap_activations[:, 1],
        color=cluster,
    )
    fig.show()


# %%

clusters = hierarchical_clustering(
    model=model,  # type: ignore
    dataset=sgraph_dataset,
    list_of_components=list_components,
    progress_bar=False,
    threshold_factor=3.7,
)

# %%
print_clusters(clusters)


# %%


del model
clean_gpu_mem()
# print_gpu_mem("after del")  # 0 Gb
model = deepcopy(raw_model)
# print_gpu_mem("after copy")  # 0 Gb
model.to(
    "cuda"
)  # TODO; dirty trick to avoid hook leak but makes many GPU memory move, sad
# print_gpu_mem("after load")  # 0.67 Gb

clean_gpu_mem()
print_gpu_mem("before sweep")
# print_gpu_mem("before sweep")  # 0.67 Gb
_ = sweep_scrub(model, sgraph_dataset, ioi_dataset, clusters, progress_bar=False)


# %%
model.reset_hooks()
clusters2 = hierarchical_clustering(
    model=model,  # type: ignore
    dataset=sgraph_dataset,
    list_of_components=list_components,
    progress_bar=False,
    threshold_factor=0.6,
)
# %%


print_clusters(clusters2)
print(clusters == clusters2)

# %%
clusters3 = hierarchical_clustering(
    model=model,  # type: ignore
    dataset=sgraph_dataset,
    list_of_components=list_components,
    progress_bar=False,
    threshold_factor=0.6,
)

# %%
print_clusters(clusters3)

# %%

clusters = hierarchical_clustering(
    model=model,  # type: ignore
    dataset=sgraph_dataset,
    list_of_components=list_components,
    progress_bar=False,
    threshold_factor=3.7,
)

print_clusters(clusters)
# %%
