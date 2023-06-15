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
# Demo of the NanoQA dataset. The full spec of the dataset can be found at: https://docs.google.com/document/d/1UrZdrr9nbk-JxVy1982Bc2hYxA88nHdTsBkyv9_8T-w/edit
# tl;dr it's a small dataset of question about short stories generated with GPT-4. The goal is that it can be used as a contraint to understand how models
# solve realistic in-context problems. I recommend reading the doc if you want a comprehensive understanding of the dataset. In this demo, I'll reuse terms introduced in the doc.

model_name = "gpt2-small"  

model = HookedTransformer.from_pretrained(model_name, device="cuda")


# %% Creating the dataset

# The dataset is composed of two parts:

# 1. The nanostories
# There are two fixed datasets each of 100 stories genrated by GPT-4. The first datasets (`nb_variable_values=2`) has only 2 possible values for each narrative variables (e.g. only two names of characters, two city names etc.). But the values are arbitrarly combined. The second dataset has 5 possible values for each variables (`nb_variable_values=5`).

# 2. The questions
# For each narrative variables, there are 3 possible questions (e.g. three different ways to ask for the name of the main character). 

# Finally, a nanoQA prompt is created by randomly matching a question and a nanostory.

nano_qa_dataset = NanoQADataset(
    nb_samples=50,
    tokenizer=model.tokenizer,
    seed=43,
    querried_variables=[ # you can choose which variables you want to have questions for
        "character_name",
        "character_occupation",
        "city",
        # "season", #gpt2-small is not good enough to answer these questions
        # "day_time",
    ],
    nb_variable_values=2, # 2 variable to make the swap graphs more readible
)


# %%

# A NanoQADataset is a rich object that tracks all the information you need about the stories, the questions and their answer.

# The nanostories stores a json describign every characteristics of the story, its seed, even the prompt that was used to generate it with gpt-4.
pprint(nano_qa_dataset.nanostories[0])

# %% Same for questions
pprint(nano_qa_dataset.questions[0])

# %% And answers
print(nano_qa_dataset.answer_texts[0]) # the full string of the answer
print(nano_qa_dataset.answer_first_token_texts[0]) # the first token of the answer in string version. They where selected such that no two answers have the same first token.
print(nano_qa_dataset.answer_tokens[0]) # the token id of the first token of the answer


# %% Let's test gpt2-small on the dataset!

d = evaluate_model(model, nano_qa_dataset, batch_size=20)
print_performance_table(d)

for querried_feature in nano_qa_dataset.querried_variables:  # type: ignore
    assert d[f"{querried_feature}_top1_mean"] > 0.5

# %% Let's run swap graphs on it!
feature_dict = get_nano_qa_features_dict(nano_qa_dataset)
sgraph_dataset = SgraphDataset(
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

PATCHED_POSITION = "END"

sgraph = SwapGraph(
    model=model,
    tok_dataset=nano_qa_dataset.prompts_tok,
    comp_metric=comp_metric,
    batch_size=100,
    proba_edge=1.0,
    patchedComponents=[
        ModelComponent(
            position=nano_qa_dataset.word_idx[PATCHED_POSITION],
            layer=9, # a name mover from the IOI circuit
            head=9,
            position_label=PATCHED_POSITION,
            name="z",
        )
    ],
)
# %%
sgraph.build(verbose=False)
# %%

sgraph.compute_weights()
_ = sgraph.compute_communities()
# %%

fig = sgraph.show_html(
    title=f"{sgraph.patchedComponents[0]} patching network. {model_name} - NanoQA",  # (sigma={percentile}th percentile)
    sgraph_dataset=sgraph_dataset,
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

# Try running again by changing z for q. What do you observe?
# 9.9 seems to also plays a role of mover: attending to the right entity and copying it! 