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

model_name = "gpt2-small"  # "gpt2-small"  # "EleutherAI/Pythia-2.8b"  # "EleutherAI/gpt-neo-2.7B"  # EleutherAI/Pythia-2.8b

model = HookedTransformer.from_pretrained(model_name, device="cuda")


# %%
nano_qa_dataset = NanoQADataset(
    nb_samples=100,
    tokenizer=model.tokenizer,
    seed=43,
    querried_variables=[
        "character_name",
        "character_occupation",
        "city",
        # "season",
        # "day_time",
    ],
)

# %%
t1 = time.time()
d = evaluate_model(model, nano_qa_dataset, batch_size=20)
print_performance_table(d)
t2 = time.time()
print(t2 - t1)


# %%
feature_dict = get_nano_qa_features_dict(nano_qa_dataset)
pnet_dataset = SgraphDataset(
    tok_dataset=nano_qa_dataset.prompts_tok,
    str_dataset=nano_qa_dataset.prompts_text,
    feature_dict=feature_dict,
)

# %%
comp_metric: CompMetric = partial(
    KL_div_sim,
    position_to_evaluate=WildPosition(nano_qa_dataset.word_idx["END"], label="END"),  # type: ignore
)
# %%

# show_attn(model=model, text=nano_qa_dataset.prompts_text[13], layer=11)


# %%


PATCHED_POSITION = "END"

pnet = SwapGraph(
    model=model,
    tok_dataset=nano_qa_dataset.prompts_tok,
    comp_metric=comp_metric,
    batch_size=100,
    proba_edge=1.0,
    patchedComponents=[
        ModelComponent(
            position=nano_qa_dataset.word_idx[PATCHED_POSITION],
            layer=9,
            head=9,
            position_label=PATCHED_POSITION,
            name="z",
        )
    ],
)
# %%
pnet.build(verbose=False)
# %%

pnet.compute_weights()
_ = pnet.compute_communities()
# %%

fig = pnet.show_html(
    title=f"{pnet.patchedComponents[0]} patching network. {model_name} - NanoQA",  # (sigma={percentile}th percentile)
    sgraph_dataset=pnet_dataset,
    feature_to_show="all",
    display=False,
    recompute_positions=True,
    iterations=1000,
)
fig.update_layout(
    height=1000,
    width=1000,
)
fig.show()
# %%
plot_extra = True
if plot_extra:
    plotHistLogLog(
        pnet.all_comp_metrics, only_y_log=False, metric_name="KL divergence histogram"
    )

    plotHistLogLog(
        pnet.all_weights,
        only_y_log=False,
        metric_name="KL divergence-based graph weights",
    )
# %%


def save_hook(tensor, hook):
    cache[hook.name] = tensor.detach().to("cuda")


model.reset_hooks()
model.run_with_hooks(
    nano_qa_dataset.prompts_tok, fwd_hooks=[("blocks.14.attn.hook_z", save_hook)]
)
# %%
