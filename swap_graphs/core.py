# %%
import copy
import dataclasses

# %%
import gc
import itertools
import random
import random as rd
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, Set


import datasets
import einops
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm.auto as tqdm
import transformer_lens
import transformer_lens.utils as utils
from attrs import define, field
from jaxtyping import Float, Int
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

from typing import Protocol, Literal
import os


def dict_val_to_str(
    d: Union[Dict[str, List[str]], Dict[str, List[int]]]
) -> Dict[str, List[str]]:
    return {k: [str(x) for x in v] for k, v in d.items()}


def wrap_str(s: str, max_line_len=100):
    """Add skip line every max_line_len characters. Ensure that no word is cut in the middle."""
    words = s.split(" ")
    wrapped_str = ""
    line_len = 0
    for word in words:
        if line_len + len(word) > max_line_len:
            wrapped_str += "\n"
            line_len = 0
        wrapped_str += word + " "
        line_len += len(word) + 1
    return wrapped_str


def objects_to_unique_ids(l: list):
    values = list(set([str(x) for x in l]))
    values.sort()
    return [values.index(str(x)) for x in l], values


def objects_to_strings(l: list) -> List[str]:
    return [str(x) for x in l]


# from utils import print_gpu_mem

NOT_A_HEAD = -3249735934

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Forward declaration to avoid cyclic dependencies
    class SwapGraph:  # type: ignore
        pass


# Define types for the file


class CompMetric(Protocol):
    # Define a type for the comparison metric function. The keywords are typed by name
    def __call__(
        self,
        logits_target: Float[torch.Tensor, "batch seq vocab"],
        logits_source: Float[torch.Tensor, "batch seq vocab"],
        target_seqs: torch.Tensor,
        target_idx: Optional[List[int]] = None,
    ):
        ...


TokDataset = Float[torch.Tensor, "batch seq"]


def discrete_labels_to_idx(labels):
    unique_labels = list(set(labels))
    return [unique_labels.index(l) for l in labels]


def create_discrete_colorscale(labels: List[str]):
    unique_labels = list(set(labels))
    color_values = px.colors.qualitative.Dark24
    colorscale = []
    for i in range(len(unique_labels)):
        effective_idx = i % len(
            color_values
        )  # If there are more labels than colors, we reuse the colors
        colorscale.append(
            [effective_idx / len(unique_labels), color_values[effective_idx]]
        )
        colorscale.append(
            [(effective_idx + 1) / len(unique_labels), color_values[effective_idx]]
        )

    return colorscale


def break_long_str(labels: str, max_length=60):
    """Add <br every max_length characters"""
    return "<br>".join(
        [labels[i : i + max_length] for i in range(0, len(labels), max_length)]
    )


def print_gpu_mem(step_name=""):
    print(
        f"{step_name} ~ {np.round(torch.cuda.memory_allocated()/1e9, 2)} Go allocated on GPU."
    )


@define
class WildPosition:
    """Stores the position of a component, can either be a single int or an array of int (e.g. if each data point has correspond to a different position)."""

    position: Union[int, List[int], torch.Tensor] = field()
    label: str = field(kw_only=True)

    def positions_from_idx(self, idx: List[int]) -> List[int]:
        if isinstance(self.position, int):
            return [self.position] * len(idx)
        else:
            assert max(idx) < len(
                self.position
            ), f"Index out of range! {max(idx)} > {len(self.position)}"
            return [int(self.position[idx[i]]) for i in range(len(idx))]

    def __attrs_post_init__(self):
        if isinstance(self.position, torch.Tensor):
            assert self.position.dim() == 1
            self.position = [int(x) for x in self.position.tolist()]


@define
class ModelComponent:
    """Stores a model component (head, layer, etc.) and its position in the model.
    * q, k, v, z refers to individual heads (head should be specified).
    * resid, mlp, attn refers to the whole layer (head should not be specified)."""

    position: WildPosition = field(kw_only=True, init=False)
    layer: int = field(kw_only=True)
    name: str = field(kw_only=True)
    head: int = field(factory=lambda: NOT_A_HEAD, kw_only=True)
    hook_name: str = field(init=False)

    @name.validator  # type: ignore
    def check(self, attribute, value):
        assert value in ["q", "k", "v", "z", "resid_pre", "resid_post", "mlp", "attn"]

    def __init__(
        self,
        position: Union[int, torch.Tensor, WildPosition],
        position_label: Optional[str] = None,
        **kwargs,
    ) -> None:
        if not isinstance(position, WildPosition):
            assert position_label is not None, "You should specify a position label!"
            self.position = WildPosition(position=position, label=position_label)
        else:
            self.position = position

        self.__attrs_init__(**kwargs)  # type: ignore

    def __attrs_post_init__(self):
        if self.name in ["q", "k", "v", "z"]:
            self.hook_name = utils.get_act_name(self.name, self.layer, "a")
            assert self.head != NOT_A_HEAD, "You should specify a head number!"
        else:
            assert self.head == NOT_A_HEAD, "You should not specify a head number!"
            self.head = NOT_A_HEAD
            if self.name in ["resid_pre", "resid_post"]:
                self.hook_name = utils.get_act_name(self.name, self.layer)
            elif self.name == "mlp":
                self.hook_name = utils.get_act_name("mlp_out", self.layer)
            elif self.name == "attn":
                self.hook_name = utils.get_act_name("attn_out", self.layer)

        assert isinstance(self.position, WildPosition)

    def is_head(self):
        return self.head != NOT_A_HEAD

    def __str__(self):
        if self.is_head():
            head_str = f".h{self.head}"
        else:
            head_str = ""
        return f"{self.hook_name}{head_str}@{self.position.label}"

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))


# %%


def component_patching_hook( # TODO add test for this function
    z: Float[torch.Tensor, ""],
    hook: HookPoint,
    cache: Float[torch.Tensor, ""],
    component: ModelComponent,
    source_idx: List[int],
    target_idx: List[int],
    source_position: Optional[WildPosition] = None,
    verbose: bool = False,
) -> Float[torch.Tensor, ""]:
    """Patches the activations of a component with the cache."""
    if verbose:
        print_gpu_mem(f"patching {component.name} {component.layer} {component.head}")
        print(z.shape)
        print(cache.shape)
        
    if source_position is None:
        source_position = component.position

    if component.is_head():
        head = component.head
        for i in range(len(source_idx)):  # TODO : vectorize
            z[i, component.position.positions_from_idx(target_idx)[i], head, :] = cache[
                i, source_position.positions_from_idx(source_idx)[i], head, :
            ]
    else:
        
        for i in range(len(source_idx)):
            z[i, component.position.positions_from_idx(target_idx)[i], :] = cache[
                i, source_position.positions_from_idx(source_idx)[i], :
            ]
    return z


@define
class ActivationStore:
    """Stores the activations of a model for a given dataset (the patched dataset), and create hooks to patch the activations of a given component (head, layer, etc)."""

    model: HookedTransformer = field(kw_only=True)
    dataset: Float[torch.Tensor, "batch pos"] = field(kw_only=True)
    listOfComponents: Optional[List[ModelComponent]] = field(kw_only=True, default=None)
    force_cache_all: bool = field(kw_only=True, default=False)
    dataset_logits: Float[torch.Tensor, "batch pos vocab"] = field(init=False)
    transformerLensCache: Union[Dict[str, torch.Tensor], ActivationCache] = field(
        init=False
    )

    def compute_cache(self):
        if self.listOfComponents is None or self.force_cache_all:
            dataset_logits, cache = self.model.run_with_cache(
                self.dataset
            )  # default, but memory inneficient
        else:
            cache = {}

            def save_hook(tensor, hook):
                cache[hook.name] = tensor.detach().to("cuda")

            dataset_logits = (
                self.model.run_with_hooks(  # only cache the components we need
                    self.dataset,
                    fwd_hooks=[(c.hook_name, save_hook) for c in self.listOfComponents],
                )
            )
        self.transformerLensCache = cache
        self.dataset_logits = dataset_logits  # type: ignore

    def __attrs_post_init__(self):
        self.compute_cache()

    def getPatchingHooksByIdx(
        self,
        source_idx: List[int],
        target_idx: List[int],
        verbose: bool = False,
        list_of_components: Optional[List[ModelComponent]] = None,
    ):
        """Create a list of hook function where the cache is computed from the stored dataset cache on the indices idx."""
        assert source_idx is not None
        assert max(source_idx) < self.dataset.shape[0]
        patchingHooks = []

        if (
            list_of_components is None
        ):  # TODO : quite dirty, remove the listOfComponents attribute
            list_of_components = self.listOfComponents

        assert list_of_components is not None

        for component in list_of_components:
            patchingHooks.append(
                (
                    component.hook_name,
                    partial(
                        component_patching_hook,
                        component=component,
                        cache=self.transformerLensCache[component.hook_name][
                            source_idx
                        ],
                        source_idx=source_idx,
                        target_idx=target_idx,
                        verbose=verbose,
                    ),
                )
            )

        return patchingHooks

    def change_component_list(self, new_list):
        """Change the list of components to patch. Update the cache accordingly (only when needed)."""
        if self.listOfComponents is not None and not self.force_cache_all:
            if [c.hook_name for c in new_list] != [
                c.hook_name for c in self.listOfComponents
            ]:
                self.listOfComponents = new_list
                self.compute_cache()  # only recompute when the list changed and when the cache is partial
        self.listOfComponents = new_list


def compute_batched_weights(
    model: HookedTransformer,
    dataset: Float[torch.Tensor, "batch pos"],
    source_IDs: List[int],
    target_IDs: List[int],
    batch_size: int,
    components_to_patch: List[ModelComponent],
    comp_metric: CompMetric,
    additional_info_gathering: Optional[
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ] = None,
    verbose: bool = False,
    activation_store: Optional[ActivationStore] = None,
    progress_bar: bool = True,
):
    all_weights = []
    if activation_store is None:
        activation_store = ActivationStore(
            model=model, dataset=dataset, listOfComponents=components_to_patch
        )
    for i in tqdm.tqdm(range(0, len(target_IDs), batch_size), disable=not progress_bar):
        source_idx = source_IDs[
            i : min(i + batch_size, len(source_IDs))
        ]  # the index that will send the cache, the once by witch we pqtch
        target_idx = target_IDs[
            i : min(i + batch_size, len(target_IDs))
        ]  # The index of the datapoints that the majority of the model will run on

        target_x = dataset[target_idx]

        gc.collect()
        torch.cuda.empty_cache()
        if verbose:
            print_gpu_mem("before run_with_hooks")

        patched_logits = model.run_with_hooks(
            target_x,
            return_type="logits",
            fwd_hooks=activation_store.getPatchingHooksByIdx(
                source_idx=source_idx, target_idx=target_idx, verbose=verbose
            ),
        )

        comp_results = comp_metric(
            logits_target=activation_store.dataset_logits[target_idx],
            logits_source=patched_logits,
            target_seqs=target_x,
            target_idx=target_idx,
        )

        if additional_info_gathering is not None:  # gather facts for debugging
            additional_info_gathering(
                activation_store.dataset_logits[target_idx], patched_logits, target_x  # type: ignore
            )

        all_weights.append(comp_results)

        # print_gpu_mem("before del")
        # del patched_logits
        # model.reset_hooks()
        # gc.collect()
        # torch.cuda.empty_cache()
        # print_gpu_mem("after del")

    return torch.cat(all_weights)


def gaussian_kernel(d, sigma):
    return np.exp(-0.5 * (d / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


from networkx.algorithms import community


@define
class SgraphDataset:
    """A dataset for the swap graph. It stores the information relevant to the dataset, as well as a set of features computed from the sequence of tokens. These features will be used to compute correlation score with the communities found by the swap graph."""

    tok_dataset: TokDataset = field(kw_only=True)
    str_dataset: List[str] = field(kw_only=True)
    feature_values: Dict[str, List[int]] = field(init=False)
    feature_labels: Dict[str, List[str]] = field(init=False)
    features: List[str] = field(init=False)
    feature_ids_to_names: Dict[str, List[str]] = field(init=False, default={})

    def __init__(
        self, feature_dict: Union[Dict[str, List[int]], Dict[str, List[str]]], **kwargs
    ) -> None:
        self.__attrs_init__(**kwargs)  # type: ignore
        self.feature_values = {}

        if isinstance(feature_dict, dict):
            self.feature_labels = dict_val_to_str(feature_dict)
            for f in feature_dict.keys():
                (
                    self.feature_values[f],
                    self.feature_ids_to_names[f],
                ) = objects_to_unique_ids(feature_dict[f])

        # for f in feature_dict.keys():
        #     self.feature_values[f] = feature_dict[f](self.tok_dataset)

        self.features = list(self.feature_values.keys())

    def compute_feature_rand(self, sgraph: "SwapGraph"):
        """Compute the rand index between the Louvain communities found in the swap graph and the feature values of the dataset."""
        assert (
            sgraph.commu_labels is not None
        ), "You need to run sgraph.compute_communities() first."
        commu = [sgraph.commu_labels[i] for i in range(len(sgraph.tok_dataset))]
        return {
            "rand": {
                f: adjusted_rand_score(self.feature_values[f], commu)
                for f in self.features
            },
            "homogeneity": {
                f: homogeneity_score(self.feature_values[f], commu)
                for f in self.features
            },
            "completeness": {
                f: completeness_score(self.feature_values[f], commu)
                for f in self.features
            },
        }

    def __len__(self):
        return len(self.str_dataset)


def compute_clustering_metrics(sgraph: "SwapGraph"):
    """Compute the clustering metrics of the patching network."""
    assert (
        sgraph.commu_labels is not None
    ), "You need to run pnet.compute_communities() first."
    commu = [sgraph.commu_labels[i] for i in range(len(sgraph.tok_dataset))]

    intra_cluster = []
    exra_cluster = []

    for u, v, w in sgraph.edges:
        if sgraph.commu_labels[u] == sgraph.commu_labels[v]:
            intra_cluster.append(w)
        else:
            exra_cluster.append(w)

    return {
        "modularity": community.modularity(
            sgraph.G_show, sgraph.commu, weight="weight", resolution=1
        ),
        "intra_cluster": np.mean(intra_cluster),
        "extra_cluster": np.mean(exra_cluster),
    }


@define
class SwapGraph:
    """Stores a swap graph. Include methods to plot the graph."""

    patchedComponents: List[ModelComponent] = field(factory=list, kw_only=True)
    model: HookedTransformer = field(kw_only=True)
    tok_dataset: Float[torch.Tensor, "batch pos"] = field(kw_only=True)
    display_dataset: List[str] = field(kw_only=True, factory=list)
    comp_metric: CompMetric = field(
        kw_only=True  # the comparison metric between the logits of the patched model and the original model
    )
    proba_edge: float = field(default=0.1, kw_only=True)
    batch_size: int = field(default=256, kw_only=True)
    raw_edges: List[Tuple[int, int, float]] = field(init=False, default=None)
    edges: List[Tuple[int, int, float]] = field(init=False, default=None)
    all_comp_metrics: List[float] = field(init=False, default=None)
    all_weights: List[float] = field(init=False)
    G: nx.DiGraph = field(init=False, default=None)
    G_show: nx.DiGraph = field(init=False, default=None)
    node_positions: Dict[int, np.ndarray] = field(init=False, default=None)

    commu: List[Set[int]] = field(
        init=False, default=None
    )  # the communities of the graph
    commu_labels: Dict[int, int] = field(
        init=False, default=None
    )  # the label of the community of each node

    def build(
        self,
        additional_info_gathering: Optional[Callable] = None,
        verbose: bool = False,
        progress_bar: bool = True,
    ):
        edges = []
        source_IDs = []
        target_IDs = []
        for i in range(len(self.tok_dataset)):
            for j, y in enumerate(self.tok_dataset):
                if torch.rand(1) > self.proba_edge:
                    continue
                if i == j:
                    continue
                source_IDs.append(j)
                target_IDs.append(i)
        if verbose:
            print(f"Number of edges: {len(edges)}")

        weights = compute_batched_weights(
            self.model,
            self.tok_dataset,
            source_IDs,
            target_IDs,
            self.batch_size,
            self.patchedComponents,
            self.comp_metric,
            additional_info_gathering,
            verbose,
            progress_bar=progress_bar,
        ).tolist()

        self.raw_edges = list(
            zip(source_IDs, target_IDs, weights)
        )  # the raw edges, the ones with the output from the comparison metric. Before plotting the edges need to go through a post-processing step to get the weight of the graph.
        self.all_comp_metrics = [x[2] for x in self.raw_edges]

        self.G = nx.DiGraph()
        for i in range(len(self.tok_dataset)):
            self.G.add_node(i, label=str(i))

        for u, v, w in self.raw_edges:
            self.G.add_edge(
                u,
                v,
                weight=w,
            )

    def compute_weights(self, func: Optional[Callable[[float], float]] = None):
        """Compute the weights of the edges for network display with force-based visualization."""
        assert (
            self.raw_edges is not None
        ), "You need to build the network before computing the weights. Call build() first."

        if func is None:
            func = partial(
                gaussian_kernel,
                sigma=np.percentile(
                    self.all_comp_metrics, 25
                ),  # hard coded default to have a reasonable kernel to got from KL to graph weights
            )

        self.edges = []
        self.all_weights = []
        for source, target, comp_metric in self.raw_edges:
            self.edges.append((source, target, func(comp_metric)))
            self.all_weights.append(func(comp_metric))

        self.G_show = nx.DiGraph()

        for i in range(len(self.tok_dataset)):
            self.G_show.add_node(i, label=str(i), color="red", node_color="r")

        for u, v, w in self.edges:
            if w == 0:
                continue
            self.G_show.add_edge(
                u,
                v,
                weight=w,
                penwidth=w,
            )

    def show(  # OLD FUNCTION, DEPRECIATED. USE show_html() INSTEAD
        self,
        color_map: Callable[[torch.Tensor], Union[List[float], List[int]]],
        title,
        color_discrete: bool = False,
        with_labels: bool = False,
        recompute_positions: bool = False,
        iterations: int = 50,
        labels: Optional[Union[List[str], Dict[int, str]]] = None,
        save_path: Optional[str] = None,
    ):
        assert (
            self.edges is not None
        ), "You need to compute the weights of the edges before displaying them. Call build() and compute_weights() first."

        if color_discrete:
            cmap = plt.cm.get_cmap("tab20", 20)  # type: ignore
        else:
            cmap = plt.cm.viridis  # type: ignore

        if self.node_positions is None or recompute_positions:
            self.node_positions = nx.spring_layout(  # type: ignore
                self.G_show, k=0.5, iterations=iterations
            )

        nx.draw_networkx_nodes(  # type: ignore
            self.G_show,
            self.node_positions,
            node_color=color_map(self.tok_dataset),  # type: ignore
            node_size=20,
            cmap=cmap,
        )

        if with_labels:
            if labels is None:
                labels = {i: str(i) for i in range(len(self.tok_dataset))}
            else:
                labels = {i: labels[i] for i in range(len(self.tok_dataset))}

            nx.draw_networkx_labels(  # type: ignore
                self.G_show,
                self.node_positions,
                labels=labels,
                font_size=7,
            )
        plt.title(title)
        plt.legend()
        if save_path is not None:
            plt.savefig(save_path)

    def show_html(
        self,
        sgraph_dataset: SgraphDataset,
        save_path=None,
        display=True,
        title: Optional[str] = None,
        feature_to_show: Optional[
            Union[str, Literal["all"], List[int], List[float]]
        ] = None,
        feature_name: Optional[str] = None,
        recompute_positions: bool = False,
        iterations: int = 100,
        color_discrete=True,
        **kwargs,
    ):
        assert (
            self.edges is not None
        ), "You need to compute the weights of the edges before displaying them. Call build() and compute_weights() first."

        if self.node_positions is None or recompute_positions:
            self.node_positions = nx.spring_layout(  # type: ignore
                self.G_show, k=0.5, iterations=iterations
            )

        color_dict = (
            {}
        )  # a dictionnary storing the color of each node for various features

        if feature_to_show is None or feature_to_show == "community":
            color_dict["community"] = self.compute_communities()
        elif feature_to_show == "all":
            color_dict["community"] = self.compute_communities()
            for feature in sgraph_dataset.feature_values:
                color_dict[feature] = sgraph_dataset.feature_values[feature]

        elif isinstance(feature_to_show, str):
            assert (
                feature_to_show in sgraph_dataset.feature_values
            ), "Wrong feature name"
            color_dict[feature_to_show] = sgraph_dataset.feature_values[feature_to_show]

        else:  # can directly pass a list of values for colouring the nodes
            assert len(feature_to_show) == len(self.G_show.nodes), "Wrong feature size"
            assert feature_name is not None, "You need to provide a feature name"
            color_dict[feature_name] = feature_to_show

        node_x = []
        node_y = []
        for node, (x, y) in self.node_positions.items():
            node_x.append(x)
            node_y.append(y)

        node_text = []
        for node, adjacencies in enumerate(self.G_show.adjacency()):
            descr = ""
            descr += f"seq: {break_long_str(sgraph_dataset.str_dataset[node])}"
            descr += f"<br>node: {node}"
            descr += f"<br>commu: {self.commu_labels[node]}"
            for feature in sgraph_dataset.feature_labels:
                descr += (
                    f"<br>{feature}: {sgraph_dataset.feature_labels[feature][node]}"
                )
            node_text.append(descr)

        if title is None:
            title = f"{self.patchedComponents[0]}" + "..." * (
                len(self.patchedComponents) > 1
            )

        trace_dict = {}
        is_first = True
        for color_name, color in color_dict.items():
            node_trace = go.Scatter(
                x=node_x,
                y=node_y,
                visible=is_first,
                mode="markers",
                hoverinfo="text",
                marker=dict(
                    showscale=True,
                    colorscale="Viridis",
                    reversescale=True,
                    color=[],
                    size=10,
                    colorbar=dict(
                        thickness=15,
                        title=color_name,
                        xanchor="left",
                        titleside="right",
                        dtick=1,
                        tickvals=list(
                            range(len(sgraph_dataset.feature_ids_to_names[color_name]))
                        )
                        if color_name in sgraph_dataset.features
                        else None,
                        ticktext=[
                            wrap_str(x, max_line_len=20)
                            for x in sgraph_dataset.feature_ids_to_names[color_name]
                        ]
                        if color_name in sgraph_dataset.features
                        else None,
                    ),
                    line_width=2,
                ),
            )
            is_first = False

            node_trace.text = node_text

            if color_discrete:
                labels_idx = discrete_labels_to_idx(color)
                node_trace.marker.color = discrete_labels_to_idx(labels_idx)  # type: ignore
                node_trace.marker.colorscale = create_discrete_colorscale(labels_idx)  # type: ignore
            else:
                node_trace.marker.color = color  # type: ignore
                node_trace.marker.colorscale = "Viridis"  # type: ignore

            trace_dict[color_name] = node_trace

        fig = go.Figure()
        for trace_name, trace in trace_dict.items():
            fig.add_trace(trace)

        fig.update_layout(
            title=title,
            titlefont_size=15,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=5, l=5, r=5, t=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )

        if len(trace_dict) > 1:
            # Add dropdown
            all_false = [False] * len(trace_dict)
            bool_list_dict = {}
            for i, k in enumerate(list(trace_dict.keys())):
                bool_list_dict[k] = all_false.copy()
                bool_list_dict[k][i] = True

            fig.update_layout(
                updatemenus=[
                    dict(
                        buttons=list(
                            [
                                dict(
                                    args=[
                                        "visible",
                                        bool_list,
                                    ],
                                    label=trace_name,
                                    method="restyle",
                                )
                                for trace_name, bool_list in bool_list_dict.items()
                            ]
                        ),
                        direction="down",
                        pad={"l": 10, "t": 10},
                        showactive=True,
                        x=0.9,
                        xanchor="right",
                        y=1.05,
                        yanchor="top",
                    ),
                ]
            )

        if display:
            fig.show()

        if save_path is not None:
            save_path = os.path.join(save_path, f"{self.patchedComponents[0]}.html")
            fig.write_html(save_path)

        if not display:
            return fig

    def compute_communities(self, resolution: float = 1.0):
        """Compute the communities of the graph using the Louvain algorithm. Return a function to be used as a color map for the nodes."""
        assert (
            self.edges is not None
        ), "You need to compute the weights of the edges before displaying them. Call build() and compute_weights() first."

        self.commu = community.louvain_communities(self.G_show, resolution=resolution)
        self.commu_labels = {}
        for i, c in enumerate(self.commu):
            for n in c:
                self.commu_labels[n] = i

        return [self.commu_labels[i] for i in range(len(self.tok_dataset))]

    def load_comp_metric_edges(self, comp_metric_values: List[Tuple[int, int, float]]):
        for u, v, w in comp_metric_values:
            assert u < len(self.tok_dataset), "Wrong samples idx"
            assert v < len(self.tok_dataset), "Wrong samples idx"
            assert isinstance(w, float), "Wrong weight type"
        self.raw_edges = comp_metric_values
        self.all_comp_metrics = [w for x, y, w in self.raw_edges]


def find_important_components(
    model: HookedTransformer,
    dataset: Float[torch.Tensor, "batch pos"],
    batch_size: int,
    components_to_search: List[ModelComponent],
    comp_metric: CompMetric,
    verbose: bool = False,
    output_shape: Optional[Tuple[int, int]] = None,
    nb_samples: int = 100,
    force_cache_all: bool = False,
):
    """Got through the components_to_search one by one and find the components that leads to the most significant change in the output of the model. This can be seen as computing a random subset of size nb_samples of the weight of the swap graph for each element and choose the one with the highest average weights."""

    results = []
    activation_store = ActivationStore(
        model=model,
        dataset=dataset,
        listOfComponents=[components_to_search[0]],
        force_cache_all=force_cache_all,
    )
    target_IDs = [rd.randint(0, len(dataset) - 1) for i in range(nb_samples)]
    source_IDs = [rd.randint(0, len(dataset) - 1) for i in range(nb_samples)]

    for i in tqdm.tqdm(range(len(components_to_search))):
        component = components_to_search[i]
        activation_store.change_component_list([component])
        weights = compute_batched_weights(
            model=model,
            dataset=dataset,
            source_IDs=source_IDs,
            target_IDs=target_IDs,
            batch_size=batch_size,
            components_to_patch=[component],
            comp_metric=comp_metric,
            verbose=verbose,
            activation_store=activation_store,
            progress_bar=False,
        )

        results.append(weights)

    if output_shape is None:
        output_shape = (len(components_to_search), nb_samples)

    return results
