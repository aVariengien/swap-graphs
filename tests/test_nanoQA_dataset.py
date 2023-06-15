
# %%
import os

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
)
from swap_graphs.core import SgraphDataset, break_long_str

from tqdm import tqdm

import networkx as nx
from sklearn.metrics.cluster import adjusted_rand_score

torch.set_grad_enabled(False)

import fire
from transformer_lens import loading
from swap_graphs.utils import plotHistLogLog
import plotly.express as px
import igviz as ig


# %%

from swap_graphs.datasets.nano_qa.nano_qa_dataset import (
    NanoQADataset,
    evaluate_model,
    get_nano_qa_features_dict,
)


from swap_graphs.datasets.nano_qa.nano_qa_utils import print_performance_table


# %%

def test_rotated_dataset():
    # %%
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    
    permutation = {"character_occupation": "city", "city": "character_occupation", "season": "character_name", "character_name": "season"}
    
    dataset = NanoQADataset(
        nb_samples=1000,
        tokenizer=tokenizer,  # type: ignore
        seed=43,
        querried_variables=[
            "character_name",
            "city",
            "season",
            "character_occupation",
        ],
    )
    
    perm_dataset = dataset.permute_querried_variable(permutation)
    
    for i in range(len(dataset)):
        orig_querried_variable = dataset.questions[i]["querried_variable"]
        perm_querried_variable = permutation[orig_querried_variable]
        perm_answer = " " + dataset.nanostories[i]["seed"][perm_querried_variable]
        assert perm_answer == perm_dataset.answer_texts[i], "Permutation failed"
        assert dataset.answer_texts[i] != perm_dataset.answer_texts[i], "Permutation failed"
    
# %%


def test_question_from():
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset1 = NanoQADataset(
        nb_samples=1000,
        tokenizer=tokenizer,  # type: ignore
        seed=43,
        querried_variables=[
            "character_name",
            "city",
            "season",
            "character_occupation",
        ],
    )
    
    dataset2 = NanoQADataset(
        nb_samples=1000,
        tokenizer=tokenizer,  # type: ignore
        seed=44,
        querried_variables=[
            "character_name",
            "city",
            "season",
            "character_occupation",
        ],
    )
    
    chimera_dataset = dataset1.question_from(dataset2)
    

    for i in range(len(chimera_dataset)):
        querried_variable = dataset2.questions[i]["querried_variable"]
        chimera_answer = " " + dataset1.nanostories[i]["seed"][querried_variable]
        assert chimera_answer == chimera_dataset.answer_texts[i], "Chimera dataset failed"
