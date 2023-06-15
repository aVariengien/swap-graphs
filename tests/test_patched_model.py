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
def test_scrub_and_tr_gpt2_small():
    # %%
    xp_name = "gpt2-small-IOI-compassionate_einstein"
    xp_path = "../xp/" + xp_name
    print(os.listdir(xp_path))
    RAW_MODEL_NAME = "gpt2-small"

    MODEL_NAME = RAW_MODEL_NAME.replace("/", "-")

    ioi_dataset: IOIDataset = load_object(xp_path, "ioi_dataset.pkl")

    sgraph_dataset = SgraphDataset(
        feature_dict=get_ioi_features_dict(ioi_dataset),
        tok_dataset=ioi_dataset.prompts_toks,
        str_dataset=ioi_dataset.prompts_text,
    )  # load_object(xp_path, "sgraph_dataset.pkl")

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
    patched_model.add_hooks_scrub_by_communities()

    scrub_ld_rd = logit_diff(patched_model, ioi_dataset).item()  # type: ignore

    assert (
        abs(float(scrub_ld_rd)) < 0.7
    ), f"The scrub should reduce the logit diff {scrub_ld_rd} vs {orig_ld}"

    # Scrub with the opti commu

    model.reset_hooks()
    patched_model = PatchedModel(
        model=model, sgraph_dataset=sgraph_dataset, communities=commu
    )
    patched_model.add_hooks_scrub_by_communities()
    scrub_ld_smart = logit_diff(patched_model, ioi_dataset).item()  # type: ignore

    assert (
        abs(scrub_ld_smart - orig_ld) < 0.3
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

    patched_model.add_hooks_scrub_by_communities(
        list_of_components=[component_name_to_obj[c] for c in SIN_str]
    )

    sin_scrub_ld = logit_diff(patched_model, ioi_dataset).item()  # type: ignore

    assert (
        abs(sin_scrub_ld - orig_ld) < 0.9
    ), f"SIN scrub should stay close to the original logit diff, {sin_scrub_ld} vs {orig_ld}"

    #
    model.reset_hooks()
    patched_model.model.reset_hooks()
    # patched_model.run_moving_pieces_experiment(feature="S gender", list_of_components=[component_name_to_obj[c] for c in S_gender_neo], feature_mapping={0:[1], 1:[0]}, reset_hooks=False)

    patched_model.add_hooks_targeted_rewrite(
        feature="Order of first names",
        list_of_components=[component_name_to_obj[c] for c in SIN_str],
        feature_mapping={"ABB": ["BAB"], "BAB": ["ABB"]},
        reset_hooks=True,
    )

    #
    tr_ld = logit_diff(patched_model, ioi_dataset).item()  # type: ignore
    tr_io_prob = probs(patched_model, ioi_dataset, type="io")
    tr_s_prob = probs(patched_model, ioi_dataset, type="s")
    tr_logits = patched_model.model(ioi_dataset.prompts_tok.to("cuda"))
    print(f"Logit diff: {tr_ld}, IO prob: {tr_io_prob.item()} S prob: {tr_s_prob.item()}")  # type: ignore

    assert (
        tr_s_prob > tr_io_prob
    ), f"The S prob should be higher than the IO prob {tr_s_prob} vs {tr_io_prob}"
    assert tr_ld < -2.25, f"The logit diff should be <-3 {tr_ld}"
    # %%
    # Test the batched patching

    patched_model.model.reset_hooks()
    hook_gen_scrub = patched_model.hook_gen_scrub_by_communities(
        list_of_components=[component_name_to_obj[c] for c in SIN_str]
    )
    scrub_logits_batch = patched_model.batched_patch(
        ioi_dataset.prompts_tok, hook_gen_scrub, batch_size=13
    )
    scrub_ld_batch = logit_diff(patched_model, ioi_dataset, logits=scrub_logits_batch).item()  # type: ignore

    patched_model.model.reset_hooks()
    hooks = hook_gen_scrub(range(len(ioi_dataset)))
    for hook in hooks:
        patched_model.model.add_hook(*hook)

    scrub_logits_no_batch = patched_model.model(ioi_dataset.prompts_tok.to("cuda"))

    scrub_ld_no_batch = logit_diff(patched_model, ioi_dataset).item()  # type: ignore

    assert (
        abs(scrub_ld_no_batch - scrub_ld_batch) < 0.3
    ), f"The batched and no batch ld should be close, {scrub_ld_no_batch} vs {scrub_ld_batch}"
    assert (
        abs(scrub_ld_no_batch - sin_scrub_ld) < 0.3
    ), f"The replicated SIN scrub should stay close to the previous SIN scrub, {scrub_ld_no_batch} vs {sin_scrub_ld}"

    # %%

    hook_gen_tr = patched_model.hook_gen_targeted_rewrite(
        feature="Order of first names",
        list_of_components=[component_name_to_obj[c] for c in SIN_str],
        feature_mapping={0: [1], 1: [0]},
    )

    tr_logits = patched_model.batched_patch(
        ioi_dataset.prompts_tok, hook_gen_tr, batch_size=13
    )

    ld_tr_batched = logit_diff(patched_model, ioi_dataset, logits=tr_logits).item()  # type: ignore

    assert (
        abs(ld_tr_batched - tr_ld) < 0.3
    ), f"The batched and no batch ld should be close, {ld_tr_batched} vs {tr_ld}"

# %%
