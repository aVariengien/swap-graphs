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
from swap_graphs.datasets.nano_qa.nano_qa_dataset import (
    NanoQADataset,
    evaluate_model,
    get_nano_qa_features_dict,
)
from swap_graphs.datasets.nano_qa.nano_qa_utils import print_performance_table


torch.set_grad_enabled(False)


xp_name = "gpt2-small-z-nanoQA-xenodochial_kilby"
show_plots = True
xp_path = "../xp"
model_name = None
# %%


def mpe(
    xp_name: str,
    xp_path: str = "../xp",
    model_name: Optional[str] = None,
    stop_after_nm=False,
    show_plots=False,
):
    # %%
    path, model_name, MODEL_NAME, dataset_name = load_config(
        xp_name, xp_path, model_name  # type: ignore
    )
    assert model_name is not None
    # xp_to_load = "gpt2-small-IOI-compassionate_einstein"
    # model_name = "gpt2-small"

    # %%

    MODEL_NAME = model_name.replace("/", "-")

    path_to_plots = f"plots/nanoQA-mpe/{MODEL_NAME}-{generate_name()}"
    os.mkdir(path_to_plots)

    assert MODEL_NAME in xp_name, "Model name should be in the xp name"

    pnet_dataset = load_object(path, "pnet_dataset.pkl")
    dataset = load_object(path, "dataset.pkl")
    comp_metric = load_object(path, "comp_metric.pkl").cpu().numpy()
    mean_comp_metric = comp_metric.mean(axis=-1)
    all_pnet_data = load_object(path, "all_data.pkl")

    model = HookedTransformer.from_pretrained(
        model_name, device="cuda"
    )  # raw model to hard reset the hooks

    d = evaluate_model(model, dataset, batch_size=20)
    for querried_feature in dataset.querried_variables:  # type: ignore
        assert d[f"{querried_feature}_top1_mean"] > 0.5

    print_performance_table(d)

    end_position = WildPosition(position=dataset.word_idx["END"], label="END")

    list_components = [
        compo_name_to_object(c, end_position, model.cfg.n_heads)
        for c in all_pnet_data.keys()
    ]

    all_mlp_components = [str(c) for c in list_components if c.name == "mlp"]

    # %%

    def find_sender(
        all_pnet_data: Dict[str, Dict[str, Any]],
        comp_metric: torch.Tensor,
        sent_feature: str,
        importance_percentile: float = 0.95,
        filtering_percentile: float = 0.95,
        anti_features: Dict[str, float] = {},
        to_exclude: List[str] = [],
    ) -> List[str]:
        """
        Find the compoenent whose pnet is highly determined by the posiitonal information
        """
        senders = []
        importance_threshold = np.quantile(comp_metric, importance_percentile)
        rand_threshold = np.quantile(
            [
                all_pnet_data[c]["feature_metrics"]["rand"][sent_feature]
                for c in all_pnet_data.keys()
            ],
            filtering_percentile,
        )
        hom_threshold = 0

        for c in all_pnet_data.keys():
            if (
                (
                    all_pnet_data[c]["feature_metrics"]["rand"][sent_feature]
                    > rand_threshold
                )
                and (
                    all_pnet_data[c]["feature_metrics"]["homogeneity"][sent_feature]
                    > hom_threshold
                )
                and (c not in to_exclude)
            ):
                l, h = component_name_to_idx(c, model.cfg.n_heads)
                if comp_metric[l, h] > importance_threshold:
                    filtered = False
                    for f, v in anti_features.items():
                        if all_pnet_data[c]["feature_metrics"]["rand"][f] > v:
                            print(f"Excluding {c} because of {f}")
                            filtered = True
                            break
                    if not filtered:
                        senders.append(c)
        return senders

    queried_var_str = find_sender(
        all_pnet_data,
        comp_metric=mean_comp_metric,
        filtering_percentile=0.9,
        importance_percentile=0.7,
        sent_feature="querried_variable",
        to_exclude=all_mlp_components,
        anti_features={"answer_first_token": 0.6},
    )

    print("Querried var", queried_var_str, len(queried_var_str))

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

    queried_var = batch_name_to_obj([queried_var_str], position=end_position)[0]

    # Create the matrix
    matrix = np.zeros((model.cfg.n_layers, model.cfg.n_heads + 1))

    # Fill the matrix with the identities of the components
    head_types = []
    layers = []
    heads = []

    types = ["queried_var"]

    for idx, component_list in enumerate([queried_var]):
        for component in component_list:
            l, h = component.layer, component.head
            head_types.append(types[idx])
            layers.append(l)
            heads.append(h)
    # %%

    # %%
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

    #  Def MPE
    # 1. MPE name movers -> directly change the ouput
    # 2. MPE name movers -> change the input

    # %% Define the patched model
    pnet_commu = {
        compo_name_to_object(c, end_position, model.cfg.n_heads): all_pnet_data[c][
            "commu"
        ]
        for c in all_pnet_data.keys()
    }

    patched_model = PatchedModel(
        model=model, sgraph_dataset=pnet_dataset, communities={}
    )

    # %% 1. MPE on IO token

    # %% balseline

    # build the permuted query dataset. The label_nano_dataset contains the same prompts as the original dataset but the answer is the answer to the permuted question (never seen!)

    PERMUTATION = {
        "city": "character_name",
        "character_name": "character_occupation",
        "character_occupation": "city",
    }

    IDENTITY_PERMUTATION = {
        "city": "city",
        "character_name": "character_name",
        "character_occupation": "character_occupation",
    }

    PERMUTATION_CONTROL = {
        "city": "character_occupation",
        "character_name": "city",
        "character_occupation": "character_name",
    }

    PERMUTATION_ID = {2: [0], 0: [1], 1: [2]}

    def create_rot_dataset(
        dataset: NanoQADataset, permutation: Dict[str, str], max_size=50
    ):
        label_nano_dataset = NanoQADataset(
            nb_samples=dataset.nb_samples,
            tokenizer=model.tokenizer,  # type: ignore
            seed=44,
            querried_variables=[
                "character_name",
                "character_occupation",
                "city",
                # "season",
                # "day_time",
            ],
        )

        label_nano_dataset.prompts_text = dataset.prompts_text[:max_size]
        label_nano_dataset.prompts_tok = dataset.prompts_tok[:max_size]
        label_nano_dataset.nb_samples = max_size
        label_nano_dataset.word_idx["END"] = dataset.word_idx["END"][:max_size]

        for i in range(len(label_nano_dataset)):
            orig_querried_variable = dataset.questions[i]["querried_variable"]
            perm_querried_variable = permutation[orig_querried_variable]
            answer_str = (
                " " + dataset.nanostories[i]["seed"][perm_querried_variable]
            )  ##we add the leading space
            answer_tokens = torch.tensor(dataset.tokenizer([answer_str])["input_ids"])[
                0
            ]  # possibly multiple tokens
            label_nano_dataset.answer_tokens[i] = int(answer_tokens[0])
        return label_nano_dataset

    label_nano_dataset = create_rot_dataset(dataset, PERMUTATION)
    control_dataset = create_rot_dataset(dataset, PERMUTATION_CONTROL)

    tiny_dataset = create_rot_dataset(dataset, IDENTITY_PERMUTATION)

    # %%

    patched_model.sgraph_dataset.feature_values[
        "querried_variable"
    ] = patched_model.sgraph_dataset.feature_values["querried_variable"][:50]
    patched_model.sgraph_dataset.tok_dataset = patched_model.sgraph_dataset.tok_dataset[
        :50
    ]

    # %% 2. MPE on sender heads: goal is to change the token outputted by the network.
    # Making output S instead of IO

    def run_series_sender_MPE(percentiles: List[float]):
        sender_mpe_results = {}

        sender_mpe_results[f"baseline"] = {}
        sender_mpe_results["mpe_results"] = {}

        sender_mpe_results["n_heads"] = []
        sender_mpe_results["heads"] = []

        sender_mpe_results["mpe_results_rot"] = {}
        sender_mpe_results["mpe_results_control"] = {}

        for percentile in percentiles:
            # basic definition of sender to avoid collision with NM

            queried_var_str = find_sender(
                all_pnet_data,
                comp_metric=mean_comp_metric,
                filtering_percentile=percentile,
                importance_percentile=0.7,
                sent_feature="querried_variable",
                to_exclude=[],
                anti_features={"answer_first_token": 0.55},
            )
            queried_var_senders = batch_name_to_obj(
                [queried_var_str], position=end_position
            )[0]

            model.reset_hooks()
            d = evaluate_model(model, tiny_dataset)
            sender_mpe_results[f"baseline"][percentile] = d

            # MPE

            patched_model.run_targetted_rewrite(
                feature="querried_variable",
                list_of_components=queried_var_senders,
                feature_mapping=PERMUTATION_ID,
                reset_hooks=True,
            )

            sender_mpe_results[f"mpe_results"][percentile] = evaluate_model(
                patched_model.model, tiny_dataset, batch_size=len(tiny_dataset)
            )
            sender_mpe_results["mpe_results_rot"][percentile] = evaluate_model(
                model, label_nano_dataset, batch_size=len(tiny_dataset)
            )

            sender_mpe_results["mpe_results_control"][percentile] = evaluate_model(
                model, control_dataset, batch_size=len(tiny_dataset)
            )

            sender_mpe_results["n_heads"].append(len(queried_var_senders))
            sender_mpe_results["heads"].append(queried_var_senders)

        return sender_mpe_results

    # %%
    percentiles = list(np.linspace(0.5, 1.0, 5))
    sender_mpe_results = run_series_sender_MPE(percentiles)

    # %%
    # Get the data for each metric
    propor_heads = [
        sender_mpe_results["n_heads"][i] / (model.cfg.n_heads * model.cfg.n_layers)
        for i in range(len(percentiles))
    ]

    for metric in d.keys():
        if "mean" not in metric:
            continue
        fig = sp.make_subplots(rows=1, cols=1, subplot_titles=(metric))
        for eval in [
            "baseline",
            "mpe_results",
            "mpe_results_rot",
            "mpe_results_control",
        ]:
            data = [sender_mpe_results[eval][p][metric] for p in percentiles]
            fig.add_trace(
                go.Scatter(
                    x=propor_heads,
                    y=data,
                    name=f"{eval}",
                    mode="lines+markers" if eval == "mpe_results_rot" else "lines",
                    line=dict(width=2),
                ),
                row=1,
                col=1,
            )

        title_suffix = f""

        fig.update_layout(
            title=f"Querried Variable Sender Heads MPE Results - {metric} {title_suffix}- {MODEL_NAME}",
            showlegend=True,
            xaxis_title="#senders /total number of heads",
            yaxis_title=metric,
        )
        if show_plots:
            fig.show()
        fig.write_html(path_to_plots + f"/sender_MPE_{metric}.html")
    # %% Show number of senders

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=percentiles,
            y=[
                sender_mpe_results["n_heads"][i]
                / (model.cfg.n_heads * model.cfg.n_layers)
                for i in range(len(percentiles))
            ],
            name=f"Nb of sender (total)",
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
    #     for i, layers in enumerate(sender_mpe_results["nm_layers"]):
    #         for l in layers:
    #             all_layers[sender_key]["percentiles"].append(percentiles[i])
    #             all_layers[sender_key]["layers"].append(l)

    fig = go.Figure()

    colors = ["red", "green", "blue"]

    for i, sender_key in enumerate(["mpe_results"]):
        for p in percentiles:
            fig.add_trace(
                go.Violin(
                    x=[p] * len(sender_mpe_results[sender_key][percentiles.index(p)]),
                    y=sender_mpe_results[sender_key][percentiles.index(p)],
                    name=f"# of {sender_key.replace('_layers', '')}",
                    line=dict(width=2, color=colors[i]),
                    box_visible=True,
                    showlegend=True if p == percentiles[0] else False,
                )
            )

    fig.update_layout(
        title=f"Distribution of querried variable senders' layers vs filter percentile threshold - {MODEL_NAME}",
        showlegend=True,
        xaxis_title="Filtering Percentile threshold",
        yaxis_title="Layer",
    )
    if show_plots:
        fig.show()
    fig.write_html(path_to_plots + "/sender_layers.html")
    # %%


if __name__ == "__main__":
    fire.Fire(mpe)
