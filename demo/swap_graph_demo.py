# %%


from functools import partial

import plotly.express as px
import torch
import tqdm.auto as tqdm
from swap_graphs.core import (
    CompMetric,
    ModelComponent,
    SgraphDataset,
    SwapGraph,
    WildPosition,
)
from swap_graphs.datasets.ioi.ioi_dataset import NAMES_GENDER, IOIDataset
from swap_graphs.datasets.ioi.ioi_utils import get_ioi_features_dict
from swap_graphs.utils import KL_div_sim
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer

torch.set_grad_enabled(False)


# %% Load the model

model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")

# %% Load the dataset

ioi_dataset = IOIDataset(N=50, seed=42, nb_names=5)

# %% Create a
feature_dict = get_ioi_features_dict(ioi_dataset)
sgraph_dataset = SgraphDataset(
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


# %% compute adjusted rand index

metrics = sgraph_dataset.compute_feature_rand(sgraph)

# Plot the graph

fig = sgraph.show_html(
    title=f"{sgraph.patchedComponents[0]} swap graph. gpt2-small. Adjused Rand index with 'IO token': {metrics['rand']['IO token']:.2f} ",  # (sigma={percentile}th percentile)
    sgraph_dataset=sgraph_dataset,
    feature_to_show="all",
    display=False,
    recompute_positions=True,
    iterations=1000,
)

fig.show()
# %%
