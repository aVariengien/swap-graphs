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
    pnet_dataset: SgraphDataset,
    ioi_dataset: IOIDataset,
    classes: Dict[ModelComponent, Dict[int, int]],
    progress_bar=True,
):
    """Scrub by compunity until layer L. Do this for L=0 to nb_layers."""
    patched_model = PatchedModel(
        model=model, sgraph_dataset=pnet_dataset, communities=classes
    )

    max_L = max([c.layer for c in classes.keys()])
    compos_list = list(classes.keys())

    results = {}
    results["logit_diff"] = []
    results["io_prob"] = []
    results["s_prob"] = []

    for L in tqdm.tqdm(range(max_L), disable=not progress_bar):
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

# loading the Pnets

# xp_to_load = "gpt2-small-IOI-compassionate_einstein"  # "EleutherAI-gpt-neo-2.7B-IOI-serene_murdock" #gpt2-small-IOI-compassionate_einstein
# model_name = "gpt2-small"


def scrub(
    xp_name: str,
    xp_path: str = "../xp",
    model_name: Optional[str] = None,
):
    path, model_name, MODEL_NAME, dataset_name = load_config(
        xp_name, xp_path, model_name  # type: ignore
    )

    pnet_dataset = load_object(path, "pnet_dataset.pkl")
    ioi_dataset = load_object(path, "ioi_dataset.pkl")
    comp_metric = load_object(path, "comp_metric.pkl")
    all_pnet_data = load_object(path, "all_data.pkl")

    if hasattr(ioi_dataset, "prompts_toks"):  # for backward compatibility
        ioi_dataset.prompts_tok = ioi_dataset.prompts_toks
    # %%
    # print_gpu_mem()
    # model = HookedTransformer.from_pretrained(RAW_MODEL_NAME, device="cuda")

    # assert_model_perf_ioi(model, ioi_dataset)
    # print_gpu_mem()

    print_gpu_mem("before loading raw model")
    assert isinstance(model_name, str)
    raw_model = HookedTransformer.from_pretrained(
        model_name, device="cpu"
    )  # raw model to hard reset the hooks
    print_gpu_mem("after loading raw model")
    model = deepcopy(raw_model)
    model.to("cuda")
    print_gpu_mem("after loading model on cuda")

    assert_model_perf_ioi(model, ioi_dataset)

    # %%
    end_position = WildPosition(position=ioi_dataset.word_idx["END"], label="END")
    list_components = [
        compo_name_to_object(c, end_position, raw_model.cfg.n_heads)
        for c in all_pnet_data.keys()
    ]

    # %%
    # tuned parameters to lead to similar entropy / cluster size on gpt2-small-IOI-compassionate_einstein
    ward_threshold_factors = [0.6, 0.9, 1.55, 2.7, 3.7][::-1]
    louvain_resolutions = [0.1, 0.575, 1.0, 1.525, 2.0]
    rand_n_classes = [1, 2, 4, 9, 1e9]  #  2, 4, 9,

    # %%
    # tuning code
    # for f in rand_n_classes:
    #     print(f"Threshold factor: {f}")
    #     # clusters = hierarchical_clustering(
    #     #     model, pnet_dataset, list_components, progress_bar=False, threshold_factor=f
    #     # )
    #     # clusters = create_pnet_communities(resolution=f)
    #     clusters = create_random_communities(
    #         list_components, n_samples=len(pnet_dataset), n_classes=int(f)
    #     )
    #     print(f"Average class entropy: {average_class_entropy(clusters)}")
    #     print(f"Average cluster size: {average_cluster_size(clusters, len(pnet_dataset))}")
    #     print()

    # %%

    # %%

    run_name = generate_name()
    print(f"Experiment name: {run_name}")

    scrub_results = {}
    verbose = False

    for technique in ["ward", "random", "pnet"]:
        print(f"Technique: {technique}")
        if technique == "ward":
            clustering_params = ward_threshold_factors
        elif technique == "pnet":
            clustering_params = louvain_resolutions
        elif technique == "random":
            clustering_params = rand_n_classes
        else:
            raise ValueError(f"Unknown technique {technique}")

        scrub_results[technique] = {}
        for f in clustering_params:
            print_gpu_mem(f"Current param: {technique} - {f}")

            if technique == "ward":
                cluster_fn = partial(
                    hierarchical_clustering,
                    model=model,  # type: ignore
                    dataset=pnet_dataset,
                    list_of_components=list_components,
                    progress_bar=False,
                )
            elif technique == "pnet":
                cluster_fn = partial(
                    create_sgraph_communities,
                    model=model,
                    list_of_components=list_components,
                    dataset=pnet_dataset,
                    all_pnet_data=all_pnet_data,
                )
            elif technique == "random":
                cluster_fn = partial(
                    create_random_communities,
                    list_compos=list_components,
                    n_samples=len(pnet_dataset),
                )
            else:
                raise ValueError(f"Unknown technique {technique}")

            if technique == "ward":
                clusters = cluster_fn(threshold_factor=f)  # type: ignore
            elif technique == "pnet":
                clusters = cluster_fn(resolution=f)  # type: ignore
            elif technique == "random":
                clusters = cluster_fn(n_classes=int(f))  # type: ignore
            else:
                raise ValueError(f"Unknown technique {technique}")

            del cluster_fn
            print_gpu_mem("after clustering")

            scrub_results[technique][f] = {}
            scrub_results[technique][f]["entropy"] = average_class_entropy(clusters)
            scrub_results[technique][f]["cluster_size"] = average_cluster_size(
                clusters, n_samples=len(pnet_dataset)
            )
            if verbose:
                print_time("before load")
                print_gpu_mem()
            del model
            clean_gpu_mem()
            # print_gpu_mem("after del")  # 0 Gb
            model = deepcopy(raw_model)
            # print_gpu_mem("after copy")  # 0 Gb
            model.to(
                "cuda"
            )  # TODO; dirty trick to avoid hook leak but makes many GPU memory move, sad
            # print_gpu_mem("after load")  # 0.67 Gb
            if verbose:
                print_time("after load")
            clean_gpu_mem()
            print_gpu_mem("before sweep")
            # print_gpu_mem("before sweep")  # 0.67 Gb
            scrub_results[technique][f]["perf"] = sweep_scrub(
                model, pnet_dataset, ioi_dataset, clusters, progress_bar=False
            )
            # print_gpu_mem("after sweep")  # 1.59 Gb
            if not os.path.exists(f"scrub_results"):
                os.makedirs(f"scrub_results")
            save_object(
                scrub_results,
                path="scrub_results",
                name=f"scrub_results_{MODEL_NAME}_{run_name}.pkl",
            )
            if verbose:
                print()
                print(scrub_results[technique][f]["perf"]["logit_diff"][::-1])
                print(scrub_results[technique][f]["perf"]["io_prob"][::-1])
                print(scrub_results[technique][f]["perf"]["s_prob"][::-1])
                print(
                    f"Average class entropy: {scrub_results[technique][f]['entropy']}, Average cluster size: {scrub_results[technique][f]['cluster_size']}"
                )

    # %% Plotting

    scrub_xp_name = f"scrub_results_{MODEL_NAME}_{run_name}.pkl"

    scrub_results = load_object(path="scrub_results", name=scrub_xp_name)

    # %%

    # plot the results with plotly

    # Create a list of colors to use for each technique

    # Define a mapping from technique string to integer index
    technique_idx = {"ward": 0, "pnet": 1, "random": 2}

    line_types = {"ward": "solid", "random": "dot", "pnet": "dash"}

    # Loop through the metrics and create a plot for each one
    for metric in ["logit_diff", "io_prob", "s_prob"]:
        # Create a list of traces for each technique
        traces = []
        for technique in scrub_results:
            for i, (param, results) in enumerate(scrub_results[technique].items()):
                traces.append(
                    go.Scatter(
                        x=list(range(len(results["perf"][metric]))),
                        y=results["perf"][metric],
                        name=f"{technique} - Avg Entropy:{results['entropy']:.1f} - Avg Cluster size:{results['cluster_size']:.2f} - Param:{param}",
                        mode="lines",
                        line=dict(
                            color=px.colors.qualitative.Plotly[i],
                            dash=line_types[technique],
                        ),
                    )
                )

        # Create the layout of the plot
        layout = go.Layout(
            title=f"{metric.capitalize()} when auto-scrub at END position until layer L - {MODEL_NAME}",
            xaxis=dict(title="Layer L"),
            yaxis=dict(title=metric.capitalize()),
            legend=dict(title="Technique"),
            height=500,
            width=1000,
        )

        # Create the figure and show it
        fig = go.Figure(data=traces, layout=layout)
        # fig.show()
        if not os.path.exists(f"plots/scrub_results/{MODEL_NAME}"):
            os.makedirs(f"plots/scrub_results/{MODEL_NAME}")
        fig.write_html(
            f"plots/scrub_results/{MODEL_NAME}/scrubs_results_{MODEL_NAME}_{metric}_{run_name}.html"
        )


# %%
if __name__ == "__main__":
    fire.Fire(scrub)
