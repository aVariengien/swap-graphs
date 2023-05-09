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

import swap_graphs as sgraph

from swap_graphs.datasets.ioi.ioi_dataset import (
    NAMES_GENDER,
    IOIDataset,
    check_tokenizer,
)
from swap_graphs.datasets.ioi.ioi_utils import (
    get_ioi_features_dict,
    logit_diff,
    logit_diff_comp,
    probs,
    assert_model_perf_ioi,
)
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
    wrap_str,
    show_mtx,
)
from swap_graphs.core import SgraphDataset, SwapGraph, break_long_str

from tqdm import tqdm

import fire
import json
from typing import Literal

torch.set_grad_enabled(False)


from swap_graphs.datasets.nano_qa.nano_qa_dataset import (
    NanoQADataset,
    evaluate_model,
    get_nano_qa_features_dict,
)
from swap_graphs.datasets.nano_qa.nano_qa_utils import print_performance_table


def auto_pnet(
    model_name: str,
    head_subpart: str = "z",
    include_mlp: bool = True,
    proportion_to_pnet: float = 1.0,
    batch_size: int = 200,
    batch_size_pnet: int = 200,
    nb_sample_eval: int = 200,
    nb_datapoints_pnet: int = 100,
    xp_path: str = "../xp",
    dataset_name: Literal["IOI", "nanoQA"] = "IOI",
):
    """
    Run patching network on components of a model.

    head_subpart: subpart of the head to patch. Can be either z, q, k or v.
    include_mlp: whether to include the mlp in the patching network. It's always their output that is patched: they are not influenced by the head_subpart param.
    proportion_to_graph: proportion of the components the most important to compute pnet on.
    nb_sample: number of patching experiments for the structural step to find the important components
    xp_path: path to the folder where the results will be saved
    batch_size: batch size for building the patching network
    """
    assert dataset_name in [
        "IOI",
        "nanoQA",
    ], "dataset_name must be either IOI or nanoQA"

    COMP_METRIC = "KL"

    # %%
    model = HookedTransformer.from_pretrained(model_name, device="cuda")

    if dataset_name == "IOI":
        assert check_tokenizer(
            model.tokenizer
        ), "The tokenizer is tokenizing some word into two tokens."
        dataset = IOIDataset(
            N=nb_datapoints_pnet,
            seed=42,
            wild_template=False,
            nb_names=5,
            tokenizer=model.tokenizer,
        )
        assert_model_perf_ioi(model, dataset)

        feature_dict = get_ioi_features_dict(dataset)
        pnet_dataset = SgraphDataset(
            tok_dataset=dataset.prompts_tok,
            str_dataset=dataset.prompts_text,
            feature_dict=feature_dict,
        )

    elif (
        dataset_name == "nanoQA"
    ):  # Define the dataset, check the model performance on it and create the pnet dataset
        dataset = NanoQADataset(
            nb_samples=nb_datapoints_pnet,
            tokenizer=model.tokenizer,  # type: ignore
            seed=43,
            querried_variables=[
                "character_name",
                "character_occupation",
                "city",
                # "season",
                # "day_time",
            ],
        )

        d = evaluate_model(model, dataset, batch_size=batch_size)
        for querried_feature in dataset.querried_variables:  # type: ignore
            assert d[f"{querried_feature}_top1_mean"] > 0.5

        print_performance_table(d)

        print("Model performance on the nanoQA dataset is good")

        feature_dict = get_nano_qa_features_dict(dataset)
        pnet_dataset = SgraphDataset(
            tok_dataset=dataset.prompts_tok,
            str_dataset=dataset.prompts_text,
            feature_dict=feature_dict,
        )

    else:
        raise ValueError("Unknown dataset_name")

    # %%
    if COMP_METRIC == "KL":
        comp_metric: CompMetric = partial(
            KL_div_sim,
            position_to_evaluate=WildPosition(dataset.word_idx["END"], label="END"),  # type: ignore
        )
    elif COMP_METRIC == "LDiff":
        comp_metric: CompMetric = partial(logit_diff_comp, ioi_dataset=dataset, keep_sign=True)  # type: ignore
    else:
        raise ValueError("Unknown comp_metric")

    # %%

    PATCHED_POSITION = "END"
    components_to_search = get_components_at_position(
        position=WildPosition(
            dataset.word_idx[PATCHED_POSITION], label=PATCHED_POSITION
        ),
        nb_layers=model.cfg.n_layers,
        nb_heads=model.cfg.n_heads,
        include_mlp=include_mlp,
        head_subpart=head_subpart,
    )

    if not os.path.exists(xp_path):
        os.mkdir(xp_path)

    xp_name = (
        model_name.replace("/", "-")
        + "-"
        + head_subpart
        + "-"
        + dataset_name
        + "-"
        + generate_name(seed=int(time.clock_gettime(0)))
    )
    xp_path = os.path.join(xp_path, xp_name)
    os.mkdir(xp_path)

    fig_path = os.path.join(xp_path, "figs")
    os.mkdir(fig_path)

    print(f"Experiment name: {xp_name} -- Experiment path: {xp_path}")

    date = time.strftime("%Hh%Mm%Ss %d-%m-%Y")  # add time stamp to the experiments
    open(os.path.join(xp_path, date), "a").close()

    # %% create config file

    config = {}
    config["model_name"] = model_name
    config["head_subpart"] = head_subpart
    config["include_mlp"] = include_mlp
    config["proportion_to_pnet"] = proportion_to_pnet
    config["batch_size"] = batch_size
    config["batch_size_pnet"] = batch_size_pnet
    config["nb_sample_eval"] = nb_sample_eval
    config["nb_datapoints_pnet"] = nb_datapoints_pnet
    config["xp_path"] = xp_path
    config["xp_name"] = xp_name
    config["dataset_name"] = dataset_name
    config["COMP_METRIC"] = COMP_METRIC
    config["PATCHED_POSITION"] = PATCHED_POSITION
    config["date"] = date
    save_object(config, xp_path, "config.pkl")

    # %%
    ### Find important components by ressampling ablation

    results = find_important_components(
        model=model,
        dataset=dataset.prompts_tok,
        nb_samples=nb_sample_eval,
        batch_size=batch_size,
        comp_metric=comp_metric,
        components_to_search=components_to_search,
        verbose=False,
        output_shape=(model.cfg.n_layers, model.cfg.n_heads + 1),
        force_cache_all=False,  # if true, will cache all the results in memory, faster but more memory intensive
    )
    if include_mlp:
        sec_dim = model.cfg.n_heads + 1
    else:
        sec_dim = model.cfg.n_heads

    save_object(
        torch.cat(results).reshape(model.cfg.n_layers, sec_dim, nb_sample_eval),
        xp_path,
        "comp_metric.pkl",
    )
    # %%

    mean_results = (
        torch.cat(results)
        .reshape(model.cfg.n_layers, sec_dim, nb_sample_eval)
        .mean(2)
        .cpu()
    )

    try:
        show_mtx(
            mean_results,
            title=f"Average component importance {model_name} on {dataset_name} at {PATCHED_POSITION}",
            nb_heads=model.cfg.n_heads,
            display=False,
            save_path=fig_path,
            color_map_label="Avg. KL div. after uniform resampling",
        )
    except:
        print("Could not save figure")

    nb_component_to_pnet = int(len(components_to_search) * proportion_to_pnet)
    important_idx = mean_results.flatten().argsort()
    sorted_components = [components_to_search[i] for i in important_idx][::-1]
    important_components = sorted_components[:nb_component_to_pnet]

    print(f"Number of components for pnet: {len(important_components)}")

    # %%

    # %%

    save_object(pnet_dataset, xp_path, "pnet_dataset.pkl")
    save_object(dataset, xp_path, "dataset.pkl")

    all_data = {}
    for i in tqdm(range(len(important_components))):
        c = important_components[i]
        pnet = SwapGraph(
            model=model,
            tok_dataset=dataset.prompts_tok,
            comp_metric=comp_metric,
            batch_size=batch_size_pnet,
            proba_edge=1.0,
            patchedComponents=[c],
        )
        pnet.build(verbose=False, progress_bar=False)
        pnet.compute_weights()
        pnet.compute_communities()

        component_data = {}
        component_data["clustering_metrics"] = compute_clustering_metrics(pnet)
        component_data["feature_metrics"] = pnet_dataset.compute_feature_rand(pnet)
        component_data["pnet_edges"] = pnet.raw_edges
        component_data["commu"] = pnet.commu_labels

        # deepcopy the component data
        all_data[str(c)] = deepcopy(component_data)

        # create html plot for the graph
        largest_rand_feature, max_rand_idx = max(
            component_data["feature_metrics"]["rand"].items(), key=lambda x: x[1]
        )
        title = wrap_str(
            f"<b>{pnet.patchedComponents[0]}</b> Average CompMetric: {np.mean(pnet.all_comp_metrics):.2f} (#{sorted_components.index(c)}), Rand idx commu-{largest_rand_feature}: {max_rand_idx:.2f}, modularity: {component_data['clustering_metrics']['modularity']:.2f}",
            max_line_len=70,
        )

        pnet.show_html(
            pnet_dataset,
            feature_to_show="all",
            title=title,
            display=False,
            save_path=fig_path,
            color_discrete=True,
        )
        if i % 10 == 0:  # save every 10 iterations
            save_object(all_data, xp_path, "all_data.pkl")
    save_object(all_data, xp_path, "all_data.pkl")


if __name__ == "__main__":
    fire.Fire(auto_pnet)
# %%
