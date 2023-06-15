# %%

import gc
import itertools
import random
import random as rd
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union


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

from typing import Protocol
import os


from swap_graphs.core import (
    SgraphDataset,
    SwapGraph,
    WildPosition,
    ModelComponent,
    CompMetric,
    ActivationStore,
    find_important_components,
    compute_clustering_metrics,
    NOT_A_HEAD,
)

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
    compo_name_to_type,
    compo_name_to_object,
)
from swap_graphs.datasets.ioi.ioi_utils import (
    get_ioi_features_dict,
    logit_diff,
    logit_diff_comp,
    probs,
)

torch.set_grad_enabled(False)



def randomize_accross_classes(
    target_idx: List[int],
    classes_list: List[int],
    classes_mapping: Dict[int, List[int]],
):
    """classes_list is a list of class id. classes_mapping maps from class_id to list of possible class_idx. Classes can be communities or features values."""
    assert type(classes_list) == list, "classes_list should be a list"
    assert (
        type(classes_mapping) == dict
    ), "classes_mapping should be a dict"  # the values in classes does need to be in [1, len(classes_mapping)]. They are arbitrary int identifying the class.

    source_idx = []
    class_to_idx = {}
    for class_id in set(classes_list):
        class_to_idx[class_id] = []
    for idx, class_id in enumerate(classes_list):
        class_to_idx[class_id].append(idx)

    new_mapping = {}
    source_idx = []
    for x in target_idx:
        new_class = random.choice(classes_mapping[classes_list[x]])
        source_idx.append(random.choice(class_to_idx[new_class]))

    return source_idx


def randomize_inside_class(target_idx: List[int], classes_dict: Dict[int, int]):
    """classes is a dict sample idex -> class id. Classes can be communities or features values."""
    assert type(classes_dict) == dict, "classes_dict should be a dict"
    source_idx = []
    class_to_idx = {}
    for class_id in set(classes_dict.values()):
        class_to_idx[class_id] = []
    for idx, class_id in classes_dict.items():
        class_to_idx[class_id].append(idx)
    for x in target_idx:
        source_idx.append(random.choice(class_to_idx[classes_dict[x]]))
    return source_idx


def randomize_matching_classes(
    target_idx: List[int], classes_list: List[int], classes_to_match_list: List[int]
):
    """Generate a source_idx such that the classes of the target_idx are matched with the classes of the classes_to_match_list. Classes can be communities or features values."""
    assert type(classes_list) == list, "classes_list should be a list"
    assert type(classes_to_match_list) == list, "classes_list should be a list"
    assert len(classes_list) == len(
        classes_to_match_list
    ), "classes_list and classes_to_match_list should have the same length"
    assert target_idx == [
        i for i in range(len(classes_list))
    ], "target_idx should be a list of int from 0 to len(classes_list)-1"

    source_idx = []

    new_class_to_idx = {}
    for class_id in set(classes_to_match_list):
        new_class_to_idx[class_id] = []
    for idx, class_id in enumerate(classes_to_match_list):
        new_class_to_idx[class_id].append(idx)

    for x in target_idx:
        orig_class = classes_list[x]
        source_idx.append(random.choice(new_class_to_idx[orig_class]))
    return source_idx

HookGenerator = Callable[
    [List[int]], List[Tuple[str, Callable]]
]  # a function that takes a list of input idx and returns a list of (component_name, hook_fn) tuples

FeatureMapping = Union[
    Dict[int, List[int]], Dict[str, List[str]]
]  # can specify the mapping with ID or with feature values


@define
class PatchedModel:
    model: HookedTransformer = field(kw_only=True)
    sgraph_dataset: SgraphDataset = field(kw_only=True)
    communities: Dict[ModelComponent, Dict[int, int]] = field(kw_only=True)
    activation_store: ActivationStore = field(init=False)
    force_cache_all = field(default=False, kw_only=True)

    def __attrs_post_init__(self):
        self.activation_store = ActivationStore(
            listOfComponents=[], # empty list: by default the cache is empty (if force_cache_all = False) and we'll populate it when generating hooks
            model=self.model,
            dataset=self.sgraph_dataset.tok_dataset,
            force_cache_all=self.force_cache_all,
        )  # the activation store is initialized with an empty list of components, we'll define the components each time we ask for patching hooks

    def batched_patch(
        self,
        x: torch.Tensor,
        hook_gen: Callable[[List[int]], List[Tuple[str, Callable]]],
        batch_size: int = 20,
        reset_hooks: bool = True,
    ) -> torch.Tensor:
        """Patch the model with the hooks returned by hook_gen and run the model on x in batches of size batch_size. Returns the logits of the patched model."""
        assert len(x.shape) == 2, "x should be a 2D tensor"

        if reset_hooks:
            self.model.reset_hooks()

        all_logits = []
        for i in range(0, len(x), batch_size):
            target_idx = list(
                range(i, min(i + batch_size, len(x)))
            )  # The index of the batch inputs
            target_x = x[i : min(i + batch_size, len(x))]

            hooks = hook_gen(target_idx)

            gc.collect()
            torch.cuda.empty_cache()

            patched_logits = self.model.run_with_hooks(
                target_x,
                return_type="logits",
                fwd_hooks=hooks,
            )
            all_logits.append(patched_logits)

        return torch.cat(all_logits)

    def hook_gen_scrub_by_communities(
        self,
        list_of_components: Optional[List[ModelComponent]] = None,
    ) -> HookGenerator:
        """Generate a hook generator that ressample input by preserving the community they are part of. The hook generator can be used to patch the model."""
        if list_of_components is None:
            list_of_components = list(self.communities.keys())

        self.activation_store.change_component_list(list_of_components)
        def hook_gen(target_idx: List[int]):
            all_hooks = []
            for component in list_of_components:
                source_idx = randomize_inside_class(
                    target_idx, self.communities[component]
                )  # the source index is the list of all the index of the dataset, but randomized within the community of the component
                hook_list = self.activation_store.getPatchingHooksByIdx(
                    source_idx=source_idx,
                    target_idx=target_idx,
                    list_of_components=[component],
                )  # we add the hook to the activation store
                all_hooks.append((hook_list[0][0], hook_list[0][1]))
            return all_hooks

        return hook_gen

    def feature_mapping_to_id(
        self,
        feature: str,
        feature_mapping: FeatureMapping,
    ) -> Dict[int, List[int]]:
        """Convert a feature mapping from feature values (or ids) to feature ids."""
        if (
            type(list(feature_mapping.keys())[0]) == int
        ):  # nothing ot do if the feature mapping is already in ids
            return feature_mapping
        else:
            feature_mapping_ids = {}
            for key, values in feature_mapping.items():
                feature_mapping_ids[
                    self.sgraph_dataset.feature_ids_to_names[feature].index(key)
                ] = [
                    self.sgraph_dataset.feature_ids_to_names[feature].index(value)
                    for value in values
                ]
            return feature_mapping_ids

    def hook_gen_targeted_rewrite(
        self,
        feature,
        list_of_components: List[ModelComponent],
        feature_mapping: Optional[FeatureMapping],
        feature_to_match: Optional[str] = None,
    ) -> HookGenerator:
        """Generate a hook generator for targeted rewrite."""
        assert (
            feature in self.sgraph_dataset.features
        ), f"The feature {feature} is not in the dataset"

        if feature_mapping is not None:
            feature_mapping_id = self.feature_mapping_to_id(feature, feature_mapping)
            
        self.activation_store.change_component_list(list_of_components)
        print("components:", list(self.activation_store.transformerLensCache.keys()))

        def hook_gen(target_idx: List[int]):
            all_hooks = []
            for component in list_of_components:
                if feature_to_match is None:
                    source_idx = randomize_accross_classes(
                        target_idx,
                        self.sgraph_dataset.feature_values[feature],
                        feature_mapping_id,
                    )  # the source index is the list of all the index of the dataset, but randomized within the community of the component
                else:
                    source_idx = randomize_matching_classes(
                        target_idx,
                        self.sgraph_dataset.feature_values[feature],
                        self.sgraph_dataset.feature_values[feature_to_match],
                    )
                hook_list = self.activation_store.getPatchingHooksByIdx(
                    source_idx=source_idx,
                    target_idx=target_idx,
                    list_of_components=[component],
                )  # we add the hook to the activation store
                all_hooks.append(
                    (hook_list[0][0], hook_list[0][1])
                )  # we add the hook to the list of hooks
            return all_hooks

        return hook_gen

    def add_hooks_scrub_by_communities(
        self,
        list_of_components: Optional[List[ModelComponent]] = None,
    ):
        """Add a hook on each of the component specified in list_of_components (or all component that have a community when None). The hook randomize the input of each component so it's run on a radnom sample _within_ the community of the component. No batching is applied"""
        hook_gen = self.hook_gen_scrub_by_communities(list_of_components)
        self.model.reset_hooks()
        target_idx = [
            i for i in range(len(self.sgraph_dataset.tok_dataset))
        ]  # the target index is the list of all the index of the dataset
        hooks = hook_gen(target_idx)
        for hook in hooks:
            self.model.add_hook(*hook)

    def add_hooks_targeted_rewrite(
        self,
        feature,
        list_of_components: List[ModelComponent],
        feature_mapping: Optional[FeatureMapping],
        reset_hooks: bool = True,
        feature_to_match: Optional[str] = None,
    ):
        """Add hooks to the model to perform a moving pieces experiment. Each element is run on inputs with feature values given by the feature_mapping. E.g. if the input x has feature value 6, the alternative input will be randomly sampled among inputs with feature values in feature_mapping[6].
        If the feature is "community", feature correspond to the communities index for each component.

        If feature_to_match is set, the feature mapping will be ignored and the source idx will be generated such that the feature to match is the same as the original feature for the target and source idx.
        """
        hook_gen = self.hook_gen_targeted_rewrite(
            feature,
            list_of_components,
            feature_mapping,
            feature_to_match,
        )
        if reset_hooks:
            self.model.reset_hooks()
        target_idx = [i for i in range(len(self.sgraph_dataset.tok_dataset))]
        hooks = hook_gen(target_idx)
        for hook in hooks:
            self.model.add_hook(*hook)

    def forward(self, input, **kwargs):
        assert torch.allclose(
            input, self.sgraph_dataset.tok_dataset
        ), "The input of the patched model should be the embedded dataset"
        return self.model(input, **kwargs)

    def __call__(self, input, **kwargs):
        return self.forward(input, **kwargs)

# %%
