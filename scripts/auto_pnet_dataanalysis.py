# %%
import copy
import dataclasses
import itertools
import os
import pickle
import random
import random as rd
import time
from copy import deepcopy
from functools import partial
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import circuitsvis as cv
import datasets
import einops
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm.auto as tqdm
import transformer_lens
import transformer_lens.utils as utils
from attrs import define, field

from swap_graphs.datasets.ioi.ioi_dataset import NAMES_GENDER, IOIDataset
from swap_graphs.datasets.ioi.ioi_utils import (
    get_ioi_features_dict,
    logit_diff,
    logit_diff_comp,
    probs,
)
from swap_graphs.datasets.nano_qa.nano_qa_dataset import get_nano_qa_features_dict
from IPython import get_ipython  # type: ignore
from jaxtyping import Float, Int
from names_generator import generate_name
from swap_graphs.core import (
    ActivationStore,
    CompMetric,
    ModelComponent,
    SwapGraph,
    WildPosition,
    find_important_components,
    SgraphDataset,
    compute_clustering_metrics,
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
from transformer_lens.loading_from_pretrained import OFFICIAL_MODEL_NAMES
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from swap_graphs.utils import (
    KL_div_sim,
    L2_dist,
    L2_dist_in_context,
    imshow,
    line,
    plotHistLogLog,
    print_gpu_mem,
    save_object,
    scatter,
    show_attn,
    get_components_at_position,
    load_object,
    show_mtx,
    component_name_to_idx,
    load_config,
)
from swap_graphs.core import SgraphDataset, SwapGraph, break_long_str

from tqdm import tqdm

import networkx as nx
from sklearn.metrics.cluster import adjusted_rand_score

torch.set_grad_enabled(False)

import fire
from transformer_lens import loading

import plotly.express as px
import igviz as ig

# %%

xp_name = "gpt2-small-z-nanoQA-xenodochial_kilby"
xp_path = "../xp"
model_name = None
# model_name = "EleutherAI/gpt-neo-2.7B"

# %%


def plot_data(
    xp_name: str,
    xp_path: str = "../xp",
    model_name: Optional[str] = None,
    show_fig=False,
    exclude_mlp_zero=False,
):
    # %%
    path, model_name, MODEL_NAME, dataset_name = load_config(
        xp_name, xp_path, model_name
    )

    pnet_dataset = load_object(path, "pnet_dataset.pkl")
    comp_metric = load_object(path, "comp_metric.pkl")
    all_pnet_data = load_object(path, "all_data.pkl")
    # if xp_name == "gpt2-small-z-nanoQA-xenodochial_kilby":
    #     dataset = load_object(path, "dataset.pkl")
    #     feature_dict = get_nano_qa_features_dict(dataset)
    #     pnet_dataset = PnetDataset(
    #         tok_dataset=dataset.prompts_tok,
    #         str_dataset=dataset.prompts_text,
    #         feature_dict=feature_dict,
    #     )
    #     print("DONNNE")
    #     print(pnet_dataset.features)

    #     for c in all_pnet_data:
    #         pnet = PatchingNetwork(
    #             model=None,
    #             tok_dataset=dataset.prompts_tok,
    #             comp_metric=None,
    #             batch_size=0,
    #             proba_edge=1.0,
    #             patchedComponents=[c],
    #         )
    #         pnet.commu_labels = all_pnet_data[c]["commu"]
    #         all_pnet_data[c]["feature_metrics"] = pnet_dataset.compute_feature_rand(
    #             pnet
    #         )
    #     save_object(all_pnet_data, path, "all_data.pkl")
    #     save_object(pnet_dataset, path, "pnet_dataset.pkl")

    #     print("DONNNE2")

    print(
        f"Number of saved pnet-ed components: {len(all_pnet_data)} / {comp_metric.shape[0]*comp_metric.shape[1]}"
    )
    # %%

    official_model_name = loading.get_official_model_name(model_name)
    cfg = loading.get_pretrained_model_config(
        model_name,
    )  # avoid loading the model weights, which is slow

    # %%
    # model = HookedTransformer.from_pretrained(RAW_MODEL_NAME, device="cuda")

    # ld = logit_diff(model, ioi_dataset)
    # io_prob = probs(model, ioi_dataset, type="io")

    # assert ld.item() > 2.5 and io_prob.item() > 0.15, "The model is not good enough on the dataset. There might be a setup problem."  # type: ignore

    # print(f"Logit diff: {ld.item()}, IO prob: {io_prob.item()}")  # type: ignore
    # %%
    mean_comp_metric = comp_metric.mean(2).cpu()
    std_comp_metric = comp_metric.std(2).cpu()

    if exclude_mlp_zero:
        mean_comp_metric[0, -1] = 0
        mean_comp_metric[-1, -1] = 0

    # %%

    if show_fig:
        show_mtx(
            mean_comp_metric,
            title=f"Mean KL divergence after random resampling {MODEL_NAME} {dataset_name}",
            color_map_label=f"KL divergence",
            nb_heads=cfg.n_heads,
            save_path=f"{path}",
            height=1000,
        )

    # %%
    if show_fig:
        plotHistLogLog(
            mean_comp_metric.flatten(),
            only_y_log=False,
            metric_name="KL divergence histogram",
        )

    # %%
    # show_attn(model, pnet_dataset.str_dataset[90], 10)

    # %% Plot modularity vs importance
    def plot_modularity_vs_importance(intra_extra_ratio=False, color="layer"):
        assert color in ["layer", "type"]
        modularities = []
        importances = []
        layers = []
        component_types = []
        for c in all_pnet_data.keys():
            l, h = component_name_to_idx(c, cfg.n_heads)
            importances.append(mean_comp_metric[l, h])
            if intra_extra_ratio:
                intra = all_pnet_data[c]["clustering_metrics"]["intra_cluster"]
                extra = all_pnet_data[c]["clustering_metrics"]["extra_cluster"]
                modularities.append(intra / extra)
            else:
                modularities.append(
                    all_pnet_data[c]["clustering_metrics"]["modularity"]
                )
            component_types.append("mlp" if h == cfg.n_heads else "attn")
            layers.append(l)

        mod_metric = (
            "Average intra cluster weight / Average extra cluster"
            if intra_extra_ratio
            else "Modularity"
        )
        fig = px.scatter(
            x=modularities,
            y=importances,
            color=layers if color == "layer" else component_types,
            hover_name=[str(c) for c in all_pnet_data.keys()],
            labels={"x": mod_metric, "y": "Importance", "color": "Layer"},
            title=f"{mod_metric} of Pnet vs Component Importance (mean KL after resampling) in {MODEL_NAME} {dataset_name}",
        )
        if show_fig:
            fig.show()

    plot_modularity_vs_importance(intra_extra_ratio=False, color="type")
    plot_modularity_vs_importance(intra_extra_ratio=False, color="layer")
    # %%

    def plot_feature_heatmap(feature: str, score_to_plot: str):
        mtx = np.zeros((cfg.n_layers, cfg.n_heads + 1))
        for c in all_pnet_data.keys():
            l, h = component_name_to_idx(c, cfg.n_heads)
            mtx[l, h] = all_pnet_data[c]["feature_metrics"][score_to_plot][feature]
        if show_fig:
            show_mtx(
                mtx,
                title=f"{feature} {score_to_plot}",
                color_map_label=f"{feature} {score_to_plot}",
                nb_heads=cfg.n_heads,
                save_path=f"{path}",
                height=800,
            )

    for metric in ["rand", "homogeneity", "completeness"]:
        for f in pnet_dataset.features:
            plot_feature_heatmap(f, metric)

    def plot_importance_feature(feature: str, score_to_plot: str, color="layer"):
        assert color in ["layer", "type"]
        importances = []
        features = []
        layers = []
        component_types = []
        for c in all_pnet_data.keys():
            l, h = component_name_to_idx(c, cfg.n_heads)
            layers.append(l)
            component_types.append("mlp" if h == cfg.n_heads else "attn")
            importances.append(mean_comp_metric[l, h])
            features.append(all_pnet_data[c]["feature_metrics"][score_to_plot][feature])
        fig = px.scatter(
            x=importances,
            y=features,
            hover_name=[str(c) for c in all_pnet_data.keys()],
            labels={
                "x": "Importance",
                "y": f"{feature} {score_to_plot}",
                "color": color,
            },
            title=f"{feature} {score_to_plot} vs Component Importance (mean KL after resampling) in {MODEL_NAME} {dataset_name}",
            color=layers if color == "layer" else component_types,
        )
        if show_fig:
            fig.show()

    for f in pnet_dataset.features:
        plot_importance_feature(f, "rand")

    # %%

    def get_nb_components_above_threshold(feature: str, score: str, threshold: float):
        all_components = list(all_pnet_data.keys())
        nb_components = 0
        for c in all_components:
            if all_pnet_data[c]["feature_metrics"][score][feature] > threshold:
                nb_components += 1
        return nb_components

    # get_nb_components_above_threshold("IO token", "rand", 0.6)

    # %%

    color_scale = px.colors.qualitative.Plotly * 10
    feature_to_color = {f: color_scale[i] for i, f in enumerate(pnet_dataset.features)}

    def metric_to_radius(metric: float, ref_metric: float):
        if metric < ref_metric:
            return 0.5
        return float(np.log(float(metric) / ref_metric) * 10)

    def create_semantic_graph(percentage_threshold=0):
        """threshold=0 means all components are included in the graph, threshold=100 means no component is included in the graph"""
        all_components = list(all_pnet_data.keys())

        G = nx.Graph()  # type: ignore

        ref_metric = np.percentile(mean_comp_metric, 50)
        importances = []
        for i in range(len(all_components)):
            c = all_components[i]
            l, h = component_name_to_idx(c, cfg.n_heads)
            importances.append(metric_to_radius(mean_comp_metric[l, h], ref_metric))

        importance_threshold = np.percentile(importances, percentage_threshold)

        for i in range(len(all_components)):
            c = all_components[i]
            l, h = component_name_to_idx(c, cfg.n_heads)

            if importances[i] > importance_threshold:
                G.add_node(
                    i,
                    label=str(all_components[i]),
                    importance=importances[i],
                    layer=l,
                    type=1 if h == cfg.n_heads else 0,
                    component=c,
                    modularity=all_pnet_data[c]["clustering_metrics"]["modularity"],
                )

                max_f = -2
                for f in pnet_dataset.features:
                    G.nodes[i][f] = all_pnet_data[c]["feature_metrics"]["rand"][f]
                    if all_pnet_data[c]["feature_metrics"]["rand"][f] > max_f:
                        max_f = all_pnet_data[c]["feature_metrics"]["rand"][f]
                        G.nodes[i]["max_rand"] = feature_to_color[f]
                        G.nodes[i]["max_rand_name"] = f
                        G.nodes[i]["max_rand_val"] = max_f

        all_weights = []

        for i in G.nodes:
            for j in G.nodes:
                if i < j:
                    ci = G.nodes[i]["component"]
                    cj = G.nodes[j]["component"]
                    comi = [
                        all_pnet_data[ci]["commu"][i]
                        for i in range(len(all_pnet_data[ci]["commu"]))
                    ]
                    comj = [
                        all_pnet_data[cj]["commu"][i]
                        for i in range(len(all_pnet_data[cj]["commu"]))
                    ]
                    commu_rand = adjusted_rand_score(comi, comj)
                    all_weights.append(commu_rand)
                    G.add_edge(
                        i, j, weight=commu_rand, penwidth=abs(commu_rand * 10 - 2)
                    )
        return G, all_weights

    if len(list(all_pnet_data.keys())) > 300:
        threshold = 70
    else:
        threshold = 0

    G, all_rand_idx = create_semantic_graph(percentage_threshold=threshold)

    # %%

    plt.hist(all_rand_idx, bins=100)
    plt.xlabel("Adjusted Rand Index between two components")
    plt.ylabel("Count")
    plt.title(
        "Distribution of Adjusted Rand Index between two components's Pnet Louvain communities"
    )

    # %%

    G_plot = G.copy()

    for u, v in G_plot.edges:
        G_plot[u][v]["opacity"] = 0.01
        if G_plot[u][v]["weight"] < 0.6:
            G_plot.remove_edge(u, v)

    # for u, v in G.edges:
    #     G[u][v]["weight"] *= G[u][v]["weight"]

    recompute_position = True
    if recompute_position:
        node_positions = nx.spring_layout(  # type: ignore
            G, k=0.5, iterations=200
        )  # computed on the complete graph G

    nx.set_node_attributes(G_plot, node_positions, "pos")  # type: ignore

    # %%

    plot_path = os.path.join("plots/semantic_graphs/", f"sematic_graphs_{xp_name}")

    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    fig = ig.plot(
        G_plot,
        transparent_background=False,
        size_method="importance",
        color_method="max_rand",
        node_text=["label", "max_rand", "max_rand_name"] + pnet_dataset.features,
        colorscale="Viridis_r",
        highlight_neighbours_on_hover=False,
        colorbar_title="Max correlated feature",
        title=f"<b>Pnet communities similarities network in {xp_name} {dataset_name}</b> <br> Node size: log importance (resampling KL) <br> Edge: Adjusted Rand Index between two components's Pnet Louvain communities",
        node_opacity=[
            min(G.nodes[i]["max_rand_val"] + 0.1, 1.0) for i in G.nodes  # type: ignore
        ],
    )

    fig.update_layout(height=1000)

    fig.write_html(os.path.join(plot_path, "semantic_graph.html"))
    print("saved in ", os.path.join(plot_path, "semantic_graph.html"))
    if show_fig:
        fig.show()
    # %%

    for color_name in ["max_rand", "layer", "type"] + pnet_dataset.features:
        fig = ig.plot(
            G_plot,
            transparent_background=False,
            size_method="importance",
            color_method=color_name,
            node_text=["label"] + pnet_dataset.features,
            colorscale="Viridis_r",
            highlight_neighbours_on_hover=False,
            colorbar_title=color_name,
            title=f"<b>Pnet communities similarities network in {xp_name} {dataset_name}</b> <br> Node size: log importance (resampling KL) <br> Edge: Adjusted Rand Index between two components's Pnet Louvain communities",
        )

        fig.update_layout(height=1000)
        if show_fig:
            fig.show()
        fig.write_html(os.path.join(plot_path, f"semantic_graph_{color_name}.html"))

    print("saved in ", os.path.join(plot_path, f"semantic_graph_{color_name}.html"))


# %%
if __name__ == "__main__":
    fire.Fire(plot_data)
