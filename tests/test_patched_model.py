# %%
from swap_graphs.PatchedModel import (
    randomize_inside_class,
    randomize_accross_classes,
    randomize_matching_classes,
    PatchedModel,
)

from swap_graphs.utils import load_object
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
    assert_model_perf_ioi,
    get_ioi_features_dict,
    logit_diff,
    logit_diff_comp,
    probs,
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
    show_mtx,
    component_name_to_idx,
    compo_name_to_object,
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

import plotly
from swap_graphs import core

from swap_graphs.datasets.ioi import ioi_dataset


def test_mappings():
    a = 0

    commu_test = {0: 42, 1: 42, 2: 42, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 2}
    source_idx_test = randomize_inside_class([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], commu_test)

    assert [commu_test[i] for i in source_idx_test] == [42, 42, 42, 1, 1, 1, 2, 2, 2, 2]
    assert source_idx_test != [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
    ], "The test failed, maybe that's bad luck, try again"

    accross_class_test = randomize_accross_classes(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [42, 42, 42, 1, 1, 1, 2, 2, 2, 2],
        {42: [1], 2: [42], 1: [2]},
    )

    assert [commu_test[i] for i in accross_class_test] == [
        1,
        1,
        1,
        2,
        2,
        2,
        42,
        42,
        42,
        42,
    ]

    alter_mapping = {0: 1, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 42, 7: 42, 8: 42, 9: 42}

    matching_class_test = randomize_matching_classes(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [42, 42, 42, 1, 1, 1, 2, 2, 2, 2],
        [1, 2, 2, 2, 2, 2, 42, 42, 42, 42],
    )

    assert [alter_mapping[i] for i in matching_class_test] == [
        42,
        42,
        42,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
    ]


# %%
def test_scrub_and_mpe_gpt2_small():
    xp_name = "gpt2-small-IOI-compassionate_einstein"
    xp_path = "../xp/" + xp_name
    print(os.listdir(xp_path))
    RAW_MODEL_NAME = "gpt2-small"

    MODEL_NAME = RAW_MODEL_NAME.replace("/", "-")

    sgraph_dataset = load_object(xp_path, "sgraph_dataset.pkl")
    ioi_dataset = load_object(xp_path, "ioi_dataset.pkl")
    if hasattr(ioi_dataset, "prompts_toks"):
        ioi_dataset.prompts_tok = ioi_dataset.prompts_toks

    comp_metric = load_object(xp_path, "comp_metric.pkl")
    all_sgraph_data = load_object(xp_path, "all_data.pkl")

    model = HookedTransformer.from_pretrained(RAW_MODEL_NAME, device="cuda")

    # test the perf

    assert_model_perf_ioi(model, ioi_dataset)

    #  define commu

    end_position = WildPosition(position=ioi_dataset.word_idx["END"], label="END")

    component_name_to_obj = {}
    for c in all_sgraph_data.keys():
        component_name_to_obj[c] = compo_name_to_object(
            c, end_position, model.cfg.n_heads
        )

    commu = {
        component_name_to_obj[c]: all_sgraph_data[c]["commu"]
        for c in all_sgraph_data.keys()
    }

    random_commus = {}
    for c in all_sgraph_data.keys():
        rd_commu = {}
        for k in commu[component_name_to_obj[c]].keys():
            rd_commu[k] = random.randint(0, 5)
        random_commus[component_name_to_obj[c]] = rd_commu.copy()

    # Test the logit diff
    model.reset_hooks()
    orig_ld = logit_diff(model, ioi_dataset)

    # Test the random commu

    patched_model = PatchedModel(
        model=model, sgraph_dataset=sgraph_dataset, communities=random_commus
    )

    non_scrub_ld = logit_diff(patched_model, ioi_dataset)

    assert non_scrub_ld == orig_ld, "The random commu should not change the logit diff"

    # Test the scrub
    patched_model.scrub_by_communities()

    scrub_ld_rd = logit_diff(patched_model, ioi_dataset).item()  # type: ignore

    assert (
        abs(float(scrub_ld_rd)) < 0.6
    ), f"The scrub should reduce the logit diff {scrub_ld_rd} vs {orig_ld}"

    # Scrub with the opti commu

    model.reset_hooks()
    patched_model = PatchedModel(
        model=model, sgraph_dataset=sgraph_dataset, communities=commu
    )
    patched_model.scrub_by_communities()
    scrub_ld_smart = logit_diff(patched_model, ioi_dataset).item()  # type: ignore

    assert (
        abs(scrub_ld_smart - orig_ld) < 0.2
    ), f"The smart scrub should stay close to the original logit diff, {scrub_ld_smart} vs {orig_ld}"

    assert (
        abs(scrub_ld_smart - orig_ld) > 0.005
    ), f"The smart scrub should not be equal to the orig, {scrub_ld_smart} vs {orig_ld}"

    #

    SIN_str = [
        "blocks.8.attn.hook_z.h6@END",
        "blocks.8.attn.hook_z.h10@END",
        "blocks.7.attn.hook_z.h3@END",
        "blocks.7.attn.hook_z.h9@END",
    ]

    #
    patched_model.model.reset_hooks()

    patched_model.scrub_by_communities(
        list_of_components=[component_name_to_obj[c] for c in SIN_str]
    )

    sin_scrub_ld = logit_diff(patched_model, ioi_dataset).item()  # type: ignore

    assert (
        abs(sin_scrub_ld - orig_ld) < 0.8
    ), f"SIN scrub should stay close to the original logit diff, {sin_scrub_ld} vs {orig_ld}"

    #
    model.reset_hooks()
    patched_model.model.reset_hooks()
    # patched_model.run_moving_pieces_experiment(feature="S gender", list_of_components=[component_name_to_obj[c] for c in S_gender_neo], feature_mapping={0:[1], 1:[0]}, reset_hooks=False)

    patched_model.run_targetted_rewrite(
        feature="Order of first names",
        list_of_components=[component_name_to_obj[c] for c in SIN_str],
        feature_mapping={0: [1], 1: [0]},
        reset_hooks=True,
    )

    #
    mpe_ld = logit_diff(patched_model, ioi_dataset).item()  # type: ignore
    mpe_io_prob = probs(patched_model, ioi_dataset, type="io")
    mpe_s_prob = probs(patched_model, ioi_dataset, type="s")
    print(f"Logit diff: {mpe_ld}, IO prob: {mpe_io_prob.item()} S prob: {mpe_s_prob.item()}")  # type: ignore

    assert (
        mpe_s_prob > mpe_io_prob
    ), f"The S prob should be higher than the IO prob {mpe_s_prob} vs {mpe_io_prob}"
    assert mpe_ld < -2.25, f"The logit diff should be <-3 {mpe_ld}"
    # %%


# %%
