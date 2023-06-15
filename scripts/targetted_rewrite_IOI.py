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
from names_generator import generate_name
import transformer_lens.utils as utils
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

import plotly.graph_objs as go
import plotly.subplots as sp


import fire

torch.set_grad_enabled(False)


# %%


def get_attention_probs(
    query_position: WildPosition,
    key_position: WildPosition,
    model: HookedTransformer,
    sgraph_dataset: SgraphDataset,
) -> torch.Tensor:
    all_attn_query_to_key = torch.zeros(model.cfg.n_layers, model.cfg.n_heads)
    logits, cache = model.run_with_cache(sgraph_dataset.tok_dataset)

    target_idx = [i for i in range(len(sgraph_dataset))]
    for l in range(model.cfg.n_layers):
        attention_pattern = cache["attn_scores", l, "attn"]
        attention_pattern = torch.softmax(attention_pattern, dim=-1)
        all_attn_query_to_key[l, :] = attention_pattern[
            range(len(sgraph_dataset)),
            :,
            query_position.positions_from_idx(target_idx),
            key_position.positions_from_idx(target_idx),
        ].mean(dim=0)
    return all_attn_query_to_key


def get_logit_lens(
    position: WildPosition, model: HookedTransformer, dataset: SgraphDataset
):
    pass  # attn.hook_result


def plot_attn_mtx(mtx):
    return px.imshow(
        mtx,
        title="Attention from END to IO",
        labels={"x": "Head", "y": "Layer", "color": "attn prob"},
        color_continuous_scale="Viridis",
        y=[str(i) for i in range(mtx.shape[0])],
        aspect="equal",
        height=600,
        width=600,
    )


xp_name = "gpt2-small-IOI-compassionate_einstein"
model_name = "gpt2-small"
xp_path = "../xp"
# %%


def tr(
    xp_name: str,
    xp_path: str = "../../xp_ioi",
    model_name: Optional[str] = None,
    stop_after_nm=False,
):
    # %%
    path, model_name, MODEL_NAME, dataset_name = load_config(
        xp_name, xp_path, model_name  # type: ignore
    )
    assert model_name is not None
    # xp_to_load = "gpt2-small-IOI-compassionate_einstein"
    # model_name = "gpt2-small"

    # %%

    show_plots = False
    MODEL_NAME = model_name.replace("/", "-")

    path_to_plots = f"plots/ioi-tr/{MODEL_NAME}-{generate_name()}"
    print(f"Saving plots to {path_to_plots}")
    os.mkdir(path_to_plots)

    assert MODEL_NAME in xp_name, "Model name should be in the xp name"

    sgraph_dataset = load_object(path, "sgraph_dataset.pkl")
    ioi_dataset = load_object(path, "ioi_dataset.pkl")
    comp_metric = load_object(path, "comp_metric.pkl").cpu().numpy()
    mean_comp_metric = comp_metric.mean(axis=-1)
    all_sgraph_data = load_object(path, "all_data.pkl")

    if hasattr(ioi_dataset, "prompts_toks"):  # for backward compatibility
        ioi_dataset.prompts_tok = ioi_dataset.prompts_toks

    model = HookedTransformer.from_pretrained(
        model_name, device="cuda"
    )  # raw model to hard reset the hooks

    assert_model_perf_ioi(model, ioi_dataset)

    end_position = WildPosition(position=ioi_dataset.word_idx["END"], label="END")
    io_position = WildPosition(position=ioi_dataset.word_idx["IO"], label="IO")
    s1_position = WildPosition(position=ioi_dataset.word_idx["S1"], label="S1")
    list_components = [
        compo_name_to_object(c, end_position, model.cfg.n_heads)
        for c in all_sgraph_data.keys()
    ]

    mtx = get_attention_probs(end_position, io_position, model, sgraph_dataset)

    fig = plot_attn_mtx(mtx)
    if show_plots:
        fig.show()
    fig.write_html(path_to_plots + "/avg_IO_attn_imshow.html")
    # %%

    def find_name_movers(
        all_sgraph_data: Dict[str, Dict[str, Any]],
        sgraph_dataset: SgraphDataset,
        end_position: WildPosition,
        io_position: WildPosition,
        comp_metric: torch.Tensor,
        to_exclude: List[str] = [],
        attn_percentile: float = 0.95,
        rand_percentile: float = 0.95,
        importance_percentile: float = 0,
    ) -> List[str]:
        """
        Find the compoenent whose i) sgraph determines non trivially by IO token ii) strong attention to IO token
        """
        name_movers = []
        importance_threshold = np.quantile(comp_metric, importance_percentile)
        if importance_percentile > 0:
            print(f"!!! Name mover importance threshold: {importance_threshold}")
        attn = get_attention_probs(end_position, io_position, model, sgraph_dataset)

        io_rands = [
            all_sgraph_data[c]["feature_metrics"]["rand"]["IO token"]
            for c in all_sgraph_data.keys()
        ]
        rand_threshold = np.quantile(io_rands, rand_percentile)
        attn_threshold = np.quantile(attn, attn_percentile)

        for c in all_sgraph_data.keys():
            if (
                all_sgraph_data[c]["feature_metrics"]["rand"]["IO token"]
                > rand_threshold
            ) and (c not in to_exclude):
                l, h = component_name_to_idx(c, model.cfg.n_heads)
                if h == model.cfg.n_heads:  # skip the mlps
                    continue
                if attn[l, h] > attn_threshold:
                    if comp_metric[l, h] > importance_threshold:
                        name_movers.append(c)

        return name_movers

    def plot_io_attn_vs_io_rand():
        io_rands = [
            all_sgraph_data[c]["feature_metrics"]["rand"]["IO token"]
            for c in all_sgraph_data.keys()
            if "mlp" not in c
        ]
        non_mlps = [c for c in all_sgraph_data.keys() if "mlp" not in c]
        attns = []
        layers = []
        for c in all_sgraph_data.keys():
            if "mlp" not in c:
                l, h = component_name_to_idx(c, model.cfg.n_heads)
                attns.append(mtx[l, h])
                layers.append(l)
        return px.scatter(
            x=io_rands,
            y=attns,
            labels={"x": "IO rand", "y": "attn to IO", "color": "Layer"},
            hover_name=non_mlps,
            color=layers,
        )

    fig = plot_io_attn_vs_io_rand()
    if show_plots:
        fig.show()
    fig.write_html(path_to_plots + "/NM_discovery_scatter_plot.html")

    # %%

    def find_sender(
        all_sgraph_data: Dict[str, Dict[str, Any]],
        comp_metric: torch.Tensor,
        sent_feature: str,
        importance_percentile: float = 0.95,
        filtering_percentile: float = 0.95,
        to_exclude: List[str] = [],
    ) -> List[str]:
        """
        Find the compoenent whose sgraph is highly determined by the posiitonal information
        """
        senders = []
        importance_threshold = np.quantile(comp_metric, importance_percentile)
        rand_threshold = np.quantile(
            [
                all_sgraph_data[c]["feature_metrics"]["rand"][sent_feature]
                for c in all_sgraph_data.keys()
            ],
            filtering_percentile,
        )
        hom_threshold = np.quantile(
            [
                all_sgraph_data[c]["feature_metrics"]["homogeneity"][sent_feature]
                for c in all_sgraph_data.keys()
            ],
            filtering_percentile,
        )

        for c in all_sgraph_data.keys():
            if (
                (
                    all_sgraph_data[c]["feature_metrics"]["rand"][sent_feature]
                    > rand_threshold
                )
                and (
                    all_sgraph_data[c]["feature_metrics"]["homogeneity"][sent_feature]
                    > hom_threshold
                )
                and (c not in to_exclude)
            ):
                l, h = component_name_to_idx(c, model.cfg.n_heads)
                if comp_metric[l, h] > importance_threshold:
                    senders.append(c)
        return senders

    pos_sender = find_sender(
        all_sgraph_data,
        comp_metric=mean_comp_metric,
        importance_percentile=0.9,
        filtering_percentile=0.95,
        sent_feature="Order of first names",
    )
    tok_sender = find_sender(
        all_sgraph_data,
        comp_metric=mean_comp_metric,
        importance_percentile=0.9,
        filtering_percentile=0.95,
        sent_feature="S1 token",
        to_exclude=pos_sender,
    )

    gender_sender = find_sender(
        all_sgraph_data,
        comp_metric=mean_comp_metric,
        importance_percentile=0.9,
        filtering_percentile=0.95,
        sent_feature="S gender",
        to_exclude=pos_sender + tok_sender,
    )

    print("Position sender", pos_sender)
    print("Token sender", tok_sender)
    print("Gender sender", gender_sender)

    name_movers = find_name_movers(
        all_sgraph_data=all_sgraph_data,
        sgraph_dataset=sgraph_dataset,
        end_position=end_position,
        io_position=io_position,
        to_exclude=pos_sender + tok_sender + gender_sender,
        comp_metric=mean_comp_metric,
        attn_percentile=0.95,
        rand_percentile=0.95,
    )

    print("Name movers", name_movers)

    def batch_name_to_obj(
        batch_name: List[List[str]], position: WildPosition
    ) -> List[List[ModelComponent]]:
        """
        Convert a list of names to a list of objects
        """
        batch_obj = []
        for name_list in batch_name:
            obj_list = []
            for name in name_list:
                obj_list.append(compo_name_to_object(name, position, model.cfg.n_heads))
            batch_obj.append(obj_list)
        return batch_obj

    pos_sender, tok_sender, gender_sender, name_movers = batch_name_to_obj(
        [pos_sender, tok_sender, gender_sender, name_movers], position=end_position
    )

    # Create the matrix
    matrix = np.zeros((model.cfg.n_layers, model.cfg.n_heads + 1))

    # Fill the matrix with the identities of the components
    head_types = []
    layers = []
    heads = []

    types = ["pos_sender", "name_movers", "tok_sender", "gender_sender"]

    for idx, component_list in enumerate(
        [
            pos_sender,
            name_movers,
            tok_sender,
            gender_sender,
        ]
    ):
        for component in component_list:
            l, h = component.layer, component.head
            head_types.append(types[idx])
            layers.append(l)
            heads.append(h)

    fig = px.scatter(
        x=heads,
        y=layers,
        color=head_types,
        labels={"x": "Head", "y": "Layer", "color": "Type"},
        height=600,
        width=800,
    )
    fig.update_traces(marker={"size": 25})

    fig.update_layout(
        yaxis_range=[-1, model.cfg.n_layers],
        xaxis_range=[-1, model.cfg.n_heads],
        title="Head types (rand percentile=0.95, attn and importance percentile=0.9)",
    )

    fig.write_html(path_to_plots + "/default_percentile_head_type.html")
    # %%

    #  Def tr
    # 1. tr name movers -> directly change the ouput
    # 2. tr name movers -> change the input

    # %% Define the patched model
    sgraph_commu = {
        compo_name_to_object(c, end_position, model.cfg.n_heads): all_sgraph_data[c][
            "commu"
        ]
        for c in all_sgraph_data.keys()
    }

    patched_model = PatchedModel(
        model=model, sgraph_dataset=sgraph_dataset, communities=sgraph_commu
    )

    # %% 1. tr on IO token

    # %% balseline

    def evaluate_model(model, ioi_dataset, all=False, head_to_compute_attn=[]):
        ld = logit_diff(model, ioi_dataset, all=all)
        io_prob = probs(model, ioi_dataset, type="io", all=all)
        s_prob = probs(model, ioi_dataset, type="s", all=all)
        if all:
            return ld.cpu().numpy(), io_prob.cpu().numpy(), s_prob.cpu().numpy()  # type: ignore
        results = {"logit diff": ld.item(), "io_prob": io_prob.item(), "s_prob": s_prob.item()}  # type: ignore

        if len(head_to_compute_attn) > 0:
            io_attn = get_attention_probs(
                end_position, io_position, model, sgraph_dataset
            )
            s_attn = get_attention_probs(
                end_position, s1_position, model, sgraph_dataset
            )
            avg_io = 0
            avg_s = 0
            for h in head_to_compute_attn:
                avg_io += io_attn[h.layer, h.head]
                avg_s += s_attn[h.layer, h.head]
            results["IO attn"] = avg_io / len(head_to_compute_attn)
            results["S attn"] = avg_s / len(head_to_compute_attn)
        return results

    def evaluate_base_rot_dataset(model, ioi_dataset, label_ioi_dataset):
        d_base = evaluate_model(model, ioi_dataset)
        d_rot = evaluate_model(model, label_ioi_dataset)
        return {"base": d_base, "rotated": d_rot}

    # %%

    def run_series_name_mover_TR(percentiles: List[float]):
        name_mover_tr_results = {}

        unique_tokens = list(
            set(ioi_dataset.io_tokenIDs[i].item() for i in range(len(ioi_dataset)))
        )

        unique_tokens.sort()

        print("Unique tokens: ", unique_tokens)

        rot_perm = {
            t: [unique_tokens[(unique_tokens.index(t) + 1) % len(unique_tokens)]]
            for t in unique_tokens
        }  # Rotate the tokens, arbitrary permutation

        rot_perm_id = {
            t: [(t + 1) % len(unique_tokens)] for t in range(len(unique_tokens))
        }  # Rotatation of the token indices in the sgraph dataset

        print("Rotated tokens: ", rot_perm)
        print("Rotated tokens id: ", rot_perm_id)

        label_ioi_dataset = deepcopy(ioi_dataset)

        label_ioi_dataset.io_tokenIDs = torch.tensor(
            [rot_perm[int(x)][0] for x in label_ioi_dataset.io_tokenIDs]
        )
        # basic definition of sender to avoid collision with NM

        pos_sender = find_sender(
            all_sgraph_data,
            comp_metric=mean_comp_metric,
            importance_percentile=0.9,
            filtering_percentile=0.95,
            sent_feature="Order of first names",
        )
        tok_sender = find_sender(
            all_sgraph_data,
            comp_metric=mean_comp_metric,
            importance_percentile=0.9,
            filtering_percentile=0.95,
            sent_feature="S1 token",
            to_exclude=pos_sender,
        )

        gender_sender = find_sender(
            all_sgraph_data,
            comp_metric=mean_comp_metric,
            importance_percentile=0.9,
            filtering_percentile=0.95,
            sent_feature="S gender",
            to_exclude=pos_sender + tok_sender,
        )
        name_mover_tr_results[f"baseline"] = {}
        name_mover_tr_results["tr_results"] = {}
        name_mover_tr_results["n_heads"] = []
        name_mover_tr_results["nm_layers"] = []

        for percentile in percentiles:
            name_movers = find_name_movers(
                comp_metric=mean_comp_metric,
                all_sgraph_data=all_sgraph_data,
                sgraph_dataset=sgraph_dataset,
                end_position=end_position,
                io_position=io_position,
                attn_percentile=percentile,
                rand_percentile=percentile,
                to_exclude=pos_sender + tok_sender + gender_sender,
            )
            name_movers = batch_name_to_obj([name_movers], position=end_position)[0]
            name_mover_tr_results["n_heads"].append(len(name_movers))
            name_mover_tr_results["nm_layers"].append([h.layer for h in name_movers])

            # baseline

            model.reset_hooks()
            d = evaluate_base_rot_dataset(model, ioi_dataset, label_ioi_dataset)
            name_mover_tr_results[f"baseline"][percentile] = d

            patched_model.add_hooks_targeted_rewrite(
                feature="IO token",
                list_of_components=name_movers,
                feature_mapping=rot_perm_id,
                reset_hooks=True,
            )
            d = evaluate_base_rot_dataset(patched_model, ioi_dataset, label_ioi_dataset)
            name_mover_tr_results["tr_results"][percentile] = d
        return name_mover_tr_results

    # %% Evaluate TR
    percentiles = list(
        np.linspace(0.8, 1.0, 10)
    )  # [0.8, 0.85, 0. 0.9, 0.95,0.975, 0.99]
    name_movers_tr_results = run_series_name_mover_TR(percentiles)

    # %%

    # Get the data for each metric
    propor_heads = [
        name_movers_tr_results["n_heads"][i] / (model.cfg.n_heads * model.cfg.n_layers)
        for i in range(len(percentiles))
    ]
    for metric in ["logit diff", "io_prob", "s_prob"]:
        fig = sp.make_subplots(rows=1, cols=1, subplot_titles=(metric))
        for eval in ["baseline", "tr_results"]:
            for rot in ["base", "rotated"]:
                data = [
                    name_movers_tr_results[eval][p][rot][metric] for p in percentiles
                ]
                fig.add_trace(
                    go.Scatter(
                        x=propor_heads,
                        y=data,
                        name=f"{eval} {rot}",
                        mode="lines+markers" if eval == "tr_results" else "lines",
                        line=dict(width=2),
                    ),
                    row=1,
                    col=1,
                )

        fig.update_layout(
            title=f"Name Movers TR Results - {metric} - {MODEL_NAME}",
            showlegend=True,
            xaxis_title="#name movers /total number of heads",
            yaxis_title=metric,
        )
        if show_plots:
            fig.show()
        fig.write_html(path_to_plots + f"/NM_TR_results_{metric}.html")

    # %%
    fig = go.Figure(
        go.Scatter(
            x=percentiles,
            y=[
                name_movers_tr_results["n_heads"][i]
                / (model.cfg.n_heads * model.cfg.n_layers)
                for i in range(len(percentiles))
            ],
            name=f"Nb of name movers",
            line=dict(width=2),
        )
    )

    fig.update_layout(
        title=f"Nb of name movers vs threshold parameter", showlegend=True
    )
    if show_plots:
        fig.show()
    fig.write_html(path_to_plots + "/NM_TR_n_heads.html")
    # %%

    thresholds = []
    all_layers = []
    for i, layers in enumerate(name_movers_tr_results["nm_layers"]):
        for l in layers:
            thresholds.append(percentiles[i])
            all_layers.append(l)

    fig = go.Figure()

    for p in percentiles:
        fig.add_trace(
            go.Violin(
                x=[p] * len(name_movers_tr_results["nm_layers"][percentiles.index(p)]),
                y=name_movers_tr_results["nm_layers"][percentiles.index(p)],
                line=dict(width=2),
                box_visible=True,
            )
        )

    fig.update_layout(
        title=f"Distribution of Name mover layers for various percentile threshold - {MODEL_NAME}",
        showlegend=False,
        xaxis_title="Percentile threshold for attn and IO tok. Rand",
        yaxis_title="Layer",
    )
    if show_plots:
        fig.show()
    fig.write_html(path_to_plots + "/NM_TR_layers.html")

    if stop_after_nm:
        return

    # %% 2. TR on sender heads: goal is to change the token outputted by the network.
    # Making output S instead of IO
    default_NM_percentile = 0.95

    def run_series_sender_TR(percentiles: List[float]):
        sender_tr_results = {}

        sender_tr_results[f"baseline"] = {}
        sender_tr_results["tr_results"] = {}

        name_movers_str = find_name_movers(
            comp_metric=mean_comp_metric,
            all_sgraph_data=all_sgraph_data,
            sgraph_dataset=sgraph_dataset,
            end_position=end_position,
            io_position=io_position,
            attn_percentile=default_NM_percentile,
            rand_percentile=default_NM_percentile,
            to_exclude=[],
        )
        name_movers = batch_name_to_obj([name_movers_str], position=end_position)[0]
        sender_tr_results["n_heads"] = []
        sender_tr_results["pos_senders_layers"] = []
        sender_tr_results["tok_senders_layers"] = []
        sender_tr_results["gender_sender_layers"] = []

        sender_tr_results["only_pos"] = {}
        sender_tr_results["only_tok"] = {}
        sender_tr_results["only_gender"] = {}
        sender_tr_results["senders_scrubbed"] = {}

        for percentile in percentiles:
            # basic definition of sender to avoid collision with NM

            pos_sender = find_sender(
                all_sgraph_data,
                comp_metric=mean_comp_metric,
                importance_percentile=0.9,
                filtering_percentile=percentile,
                sent_feature="Order of first names",
            )
            tok_sender = find_sender(
                all_sgraph_data,
                comp_metric=mean_comp_metric,
                importance_percentile=0.9,
                filtering_percentile=percentile,
                sent_feature="S1 token",
                to_exclude=pos_sender,
            )

            gender_sender = find_sender(
                all_sgraph_data,
                comp_metric=mean_comp_metric,
                importance_percentile=0.9,
                filtering_percentile=percentile,
                sent_feature="S gender",
                to_exclude=pos_sender + tok_sender,
            )
            pos_sender, tok_sender, gender_sender = batch_name_to_obj(
                [pos_sender, tok_sender, gender_sender], position=end_position
            )

            all_senders = pos_sender + tok_sender + gender_sender
            sender_tr_results["n_heads"].append(len(all_senders))
            sender_tr_results["pos_senders_layers"].append(
                [h.layer for h in pos_sender]
            )
            sender_tr_results["tok_senders_layers"].append(
                [h.layer for h in tok_sender]
            )
            sender_tr_results["gender_sender_layers"].append(
                [h.layer for h in gender_sender]
            )

            print(pos_sender, tok_sender, gender_sender)
            # baseline

            model.reset_hooks()
            d = evaluate_model(model, ioi_dataset, head_to_compute_attn=name_movers)
            sender_tr_results[f"baseline"][percentile] = d

            # TR

            patched_model.add_hooks_targeted_rewrite(
                feature="S gender",
                list_of_components=gender_sender,
                feature_mapping={},
                feature_to_match="IO gender",
                reset_hooks=True,
            )

            patched_model.add_hooks_targeted_rewrite(
                feature="Order of first names",
                list_of_components=pos_sender,
                feature_mapping={0: [1], 1: [0]},
                reset_hooks=False,
            )

            patched_model.add_hooks_targeted_rewrite(
                feature="S1 token",
                list_of_components=tok_sender,
                feature_mapping={},
                reset_hooks=False,
                feature_to_match="IO token",
            )

            d = evaluate_model(
                patched_model.model, ioi_dataset, head_to_compute_attn=name_movers
            )
            sender_tr_results[f"tr_results"][percentile] = d

            # partial TR
            patched_model.add_hooks_targeted_rewrite(
                feature="S gender",
                list_of_components=gender_sender,
                feature_mapping={},
                feature_to_match="IO gender",
                reset_hooks=True,
            )
            sender_tr_results["only_gender"][percentile] = evaluate_model(
                patched_model.model, ioi_dataset, head_to_compute_attn=name_movers
            )

            patched_model.add_hooks_targeted_rewrite(
                feature="Order of first names",
                list_of_components=pos_sender,
                feature_mapping={0: [1], 1: [0]},
                reset_hooks=True,
            )
            sender_tr_results["only_pos"][percentile] = evaluate_model(
                patched_model.model, ioi_dataset, head_to_compute_attn=name_movers
            )

            patched_model.add_hooks_targeted_rewrite(
                feature="S1 token",
                list_of_components=tok_sender,
                feature_mapping={},
                reset_hooks=True,
                feature_to_match="IO token",
            )
            sender_tr_results["only_tok"][percentile] = evaluate_model(
                patched_model.model, ioi_dataset, head_to_compute_attn=name_movers
            )

            patched_model.add_hooks_targeted_rewrite(
                feature="IO gender",  # random feature chosen
                list_of_components=tok_sender + pos_sender + gender_sender,
                feature_mapping={0: [0, 1], 1: [0, 1]},  # full scrub
                reset_hooks=True,
            )

            sender_tr_results["senders_scrubbed"][percentile] = evaluate_model(
                patched_model.model, ioi_dataset, head_to_compute_attn=name_movers
            )

        return sender_tr_results

    # %%
    percentiles = list(np.linspace(0.95, 1.0, 10))
    sender_tr_results = run_series_sender_TR(percentiles)

    # %%
    # Get the data for each metric
    propor_heads = [
        sender_tr_results["n_heads"][i] / (model.cfg.n_heads * model.cfg.n_layers)
        for i in range(len(percentiles))
    ]

    for metric in ["logit diff", "io_prob", "s_prob", "IO attn", "S attn"]:
        fig = sp.make_subplots(rows=1, cols=1, subplot_titles=(metric))
        for eval in [
            "baseline",
            "tr_results",
            "only_tok",
            "only_pos",
            "only_gender",
            "senders_scrubbed",
        ]:
            data = [sender_tr_results[eval][p][metric] for p in percentiles]
            fig.add_trace(
                go.Scatter(
                    x=propor_heads,
                    y=data,
                    name=f"{eval}",
                    mode="lines+markers" if eval == "tr_results" else "lines",
                    line=dict(width=2),
                ),
                row=1,
                col=1,
            )
        if metric in ["IO attn", "S attn"]:
            title_suffix = f"Name Mover attn (precentile={default_NM_percentile})"
        else:
            title_suffix = f""

        fig.update_layout(
            title=f"Sender Heads TR Results - {metric} {title_suffix}- {MODEL_NAME}",
            showlegend=True,
            xaxis_title="#senders /total number of heads",
            yaxis_title=metric,
        )
        if show_plots:
            fig.show()
        fig.write_html(path_to_plots + f"/sender_TR_{metric}.html")
    # %% Show number of senders

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=percentiles,
            y=[
                sender_tr_results["n_heads"][i]
                / (model.cfg.n_heads * model.cfg.n_layers)
                for i in range(len(percentiles))
            ],
            name=f"Nb of sender (total)",
            line=dict(width=2),
        )
    )

    for sender_key in [
        "gender_sender_layers",
        "pos_senders_layers",
        "tok_senders_layers",
    ]:
        fig.add_trace(
            go.Scatter(
                x=percentiles,
                y=[
                    len(sender_tr_results[sender_key][i])
                    / (model.cfg.n_heads * model.cfg.n_layers)
                    for i in range(len(percentiles))
                ],
                name=f"Nb of {sender_key.replace('_layers', '')}",
                line=dict(width=2),
            )
        )

    fig.update_layout(title=f"Nb of sender vs threshold parameter", showlegend=True)
    if show_plots:
        fig.show()
    fig.write_html(path_to_plots + "/sender_n_heads.html")

    # %% Show position of senders

    # all_layers = {}

    # for sender_key in ["gender_sender_layers", "pos_senders_layers", "tok_senders_layers"]:
    #     all_layers[sender_key] = {"layers": [], "percentiles": []}
    #     for i, layers in enumerate(sender_tr_results["nm_layers"]):
    #         for l in layers:
    #             all_layers[sender_key]["percentiles"].append(percentiles[i])
    #             all_layers[sender_key]["layers"].append(l)

    fig = go.Figure()

    colors = ["red", "green", "blue"]

    for i, sender_key in enumerate(
        ["gender_sender_layers", "pos_senders_layers", "tok_senders_layers"]
    ):
        for p in percentiles:
            fig.add_trace(
                go.Violin(
                    x=[p] * len(sender_tr_results[sender_key][percentiles.index(p)]),
                    y=sender_tr_results[sender_key][percentiles.index(p)],
                    name=f"# of {sender_key.replace('_layers', '')}",
                    line=dict(width=2, color=colors[i]),
                    box_visible=True,
                    showlegend=True if p == percentiles[0] else False,
                )
            )

    fig.update_layout(
        title=f"Distribution of senders' layers vs filter percentile threshold - {MODEL_NAME}",
        showlegend=True,
        xaxis_title="Filtering Percentile threshold",
        yaxis_title="Layer",
    )
    if show_plots:
        fig.show()
    fig.write_html(path_to_plots + "/sender_layers.html")
    # %%


if __name__ == "__main__":
    fire.Fire(tr)
