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
from swap_graphs.core import SgraphDataset, SwapGraph

from tqdm import tqdm

import networkx as nx
from sklearn.metrics.cluster import adjusted_rand_score

torch.set_grad_enabled(False)

import fire
from transformer_lens import loading

import plotly.express as px
import igviz as ig

import plotly


# %% Load the model

model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")

# %% Load the dataset

ioi_dataset = IOIDataset(N=50, seed=42, nb_names=5)

# %% Create a
feature_dict = get_ioi_features_dict(ioi_dataset)
pnet_dataset = SgraphDataset(
    tok_dataset=ioi_dataset.prompts_tok,
    str_dataset=ioi_dataset.prompts_text,
    feature_dict=feature_dict,
)

comp_metric: CompMetric = partial(
    KL_div_sim,
    position_to_evaluate=WildPosition(ioi_dataset.word_idx["END"], label="END"),  # type: ignore
)


PATCHED_POSITION = "END"

sgraph = SwapGraph(
    model=model,
    tok_dataset=ioi_dataset.prompts_tok,
    comp_metric=comp_metric,
    batch_size=300,
    proba_edge=1.0,
    patchedComponents=[
        ModelComponent(
            position=ioi_dataset.word_idx[PATCHED_POSITION],
            layer=9,
            head=9,
            position_label=PATCHED_POSITION,
            name="z",
        )
    ],
)
sgraph.build(verbose=False)
sgraph.compute_weights()
com_cmap = sgraph.compute_communities()

# check the plotting function

fig = sgraph.show_html(
    title=f"{sgraph.patchedComponents[0]} patching network. gpt2-small ",  # (sigma={percentile}th percentile)
    sgraph_dataset=pnet_dataset,
    feature_to_show="all",
    display=False,
    recompute_positions=True,
    iterations=1000,
)

metrics = pnet_dataset.compute_feature_rand(sgraph)
