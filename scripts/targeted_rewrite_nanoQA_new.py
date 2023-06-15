# %%
import gc
import itertools
import os
import random
import random as rd
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    Literal,
)

from swap_graphs.core import component_patching_hook



import datasets
import einops
import fire
import matplotlib as mpl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm.auto as tqdm
import transformer_lens
import transformer_lens.utils as utils
from attrs import define, field
from swap_graphs.communities_utils import (
    average_class_entropy,
    average_cluster_size,
    create_sgraph_communities,
    hierarchical_clustering,
)
from names_generator import generate_name
import transformer_lens.utils as utils
from swap_graphs.datasets.ioi.ioi_utils import (
    IOIDataset,
    logit_diff,
    probs,
    assert_model_perf_ioi,
)
from jaxtyping import Float, Int
from names_generator import generate_name
from swap_graphs.PatchedModel import PatchedModel
from swap_graphs.core import (
    NOT_A_HEAD,
    ActivationStore,
    CompMetric,
    ModelComponent,
    SwapGraph,
    SgraphDataset,
    TokDataset,
    WildPosition,
    break_long_str,
    compute_clustering_metrics,
    find_important_components,
)
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
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
from umap import UMAP
from swap_graphs.utils import (
    KL_div_sim,
    L2_dist,  # type: ignore
    L2_dist_in_context,
    clean_gpu_mem,
    compo_name_to_object,
    component_name_to_idx,
    create_random_communities,
    get_components_at_position,
    imshow,
    line,
    load_object,
    plotHistLogLog,
    print_gpu_mem,
    print_time,
    save_object,
    scatter,
    show_attn,
    show_mtx,
    load_config,
)

import plotly.graph_objs as go
import plotly.subplots as sp


import fire
from swap_graphs.datasets.nano_qa.nano_qa_dataset import (
    compute_random_guess,
    NanoQADataset,
    evaluate_model,
    get_nano_qa_features_dict,
)
from swap_graphs.datasets.nano_qa.nano_qa_utils import print_performance_table
from dataclasses import dataclass

torch.set_grad_enabled(False)


def steerers_from_resid(layer: int, position: WildPosition) -> List[ModelComponent]:
    return [ModelComponent(position=position, layer=layer, name="resid_post")]


def random_steerers(
    percent: float, all_components: List[ModelComponent]
) -> List[ModelComponent]:
    return random.sample(all_components, int(percent * len(all_components)))


def print_once(msg: Optional[str] = None):
    if not hasattr(print_once, "printed") or msg is None:
        print_once.printed = set()
    if msg not in print_once.printed:
        print_once.printed.add(msg)
        print(msg)


def steerers_from_sgraphs(
    all_sgraph_data: Dict[str, Dict[str, Any]],
    comp_metric_array: torch.Tensor,
    model: HookedTransformer,
    position: WildPosition,
    sent_feature: str,
    importance_percentile: float = 0.95,
    filtering_percentile: float = 0.95,
    anti_features: Dict[str, float] = {},
    filter_by_max=False,
) -> List[ModelComponent]:
    """
    Find the compoenent whose sgraph is highly determined by the posiitonal information
    """
    senders = []
    importance_threshold = np.quantile(comp_metric_array, importance_percentile)
    rand_threshold = np.quantile(
        [
            all_sgraph_data[c]["feature_metrics"]["rand"][sent_feature]
            for c in all_sgraph_data.keys()
        ],
        filtering_percentile,
    )

    print_once()

    for c in all_sgraph_data.keys():
        if all_sgraph_data[c]["feature_metrics"]["rand"][sent_feature] > rand_threshold:
            l, h = component_name_to_idx(c, model.cfg.n_heads)
            if comp_metric_array[l, h] > importance_threshold:
                # remove the component if it has an anti-feature with high rand
                filtered = False
                for f, v in anti_features.items():
                    if all_sgraph_data[c]["feature_metrics"]["rand"][f] > v:
                        print_once(f"Excluding {c} because of {f}")
                        filtered = True
                        break
                if not filtered:
                    # check if the featue with maximal rand is "sent feature"
                    max_rand_feature = max(
                        all_sgraph_data[c]["feature_metrics"]["rand"].items(),
                        key=lambda x: x[1],
                    )[0]
                    if max_rand_feature == sent_feature and filter_by_max:
                        print_once("apply select by max feature")
                        senders.append(
                            compo_name_to_object(c, position, model.cfg.n_heads)
                        )
                    else:
                        print_once("Used default branch")
                        senders.append(
                            compo_name_to_object(c, position, model.cfg.n_heads)
                        )
    return senders


def random_subset_of_steerers(
    all_sgraph_data: Dict[str, Dict[str, Any]],
    comp_metric_array: torch.Tensor,
    model: HookedTransformer,
    position: WildPosition,
    sent_feature: str,
    importance_percentile: float = 0.95,
    filtering_percentile: float = 0.95,
    anti_features: Dict[str, float] = {},
    to_exclude: List[str] = [],
    percentile_to_sample: float = 0.2,
) -> List[ModelComponent]:
    """Find a set of steerers that made 20% of the components at the last token position. Then sample a random subset of from this set suc that the subset is `percentile_to_sample` percent of the number of component at the last token position."""
    pass


@dataclass
class ExperimentConfig:
    steerer_type: Literal["random", "resid", "sgraph", "random_subset"]
    param_list: List[Any]
    name: str
    nb_replications: int = 1


@dataclass
class PermutedDataset:
    orig_dataset: NanoQADataset
    permutatiom: Dict[str, str]
    name: str

    def __post_init__(self, **kwargs):
        self.permuted_dataset = self.orig_dataset.permute_querried_variable(
            self.permutatiom
        )


def config_to_steerers(
    config: ExperimentConfig,
    all_components: List[ModelComponent],
    all_sgraph_data: Dict[str, Dict[str, Any]],
    comp_metric_array: torch.Tensor,
    model: HookedTransformer,
    position: WildPosition,
    sent_feature: str,
) -> List[List[ModelComponent]]:
    """Convert a configuration to a list of list of steerers. A list of steers for each parameter"""
    steerers = []
    if config.steerer_type == "random":
        for k in config.param_list:
            steerers.append(random_steerers(percent=k, all_components=all_components))
    elif config.steerer_type == "resid":
        for k in config.param_list:
            if k <= 0:
                steerers.append([])
            else:
                steerers.append(steerers_from_resid(layer=k, position=position))
    elif config.steerer_type == "sgraph":
        for k in config.param_list:
            steerers.append(
                steerers_from_sgraphs(
                    all_sgraph_data=all_sgraph_data,
                    comp_metric_array=comp_metric_array,
                    model=model,
                    position=position,
                    sent_feature=sent_feature,
                    importance_percentile=IMPORTANCE_PERCENTILE,
                    filtering_percentile=k,
                    anti_features={"answer_first_token": 0.55},
                )
            )
    else:
        raise ValueError(f"Unknown steerer type {config.steerer_type}")
    return steerers


ALL_METRICS = ["prob", "top1", "rank"]


def compute_steerers_stats(
    steerers: List[ModelComponent],
    model: HookedTransformer,
    config: ExperimentConfig,
    param_idx: int,
):
    total_nb_components = (model.cfg.n_heads + 1) * model.cfg.n_layers
    if config.steerer_type == "resid":
        proportion_patched = max(0, config.param_list[param_idx] / model.cfg.n_layers)
        nb_MLP = config.param_list[param_idx]
        steerer_layers = [
            layer
            for k in range(model.cfg.n_heads + 1)
            for layer in range(config.param_list[param_idx])
        ]  # patching the residual stream is patching everu component until the given layer.
    else:
        proportion_patched = len(steerers) / total_nb_components
        nb_MLP = len([s for s in steerers if not s.is_head()])
        steerer_layers = [s.layer for s in steerers]

    return proportion_patched, nb_MLP, steerer_layers


def run_steering_experiment(
    steerers_list: List[List[ModelComponent]],
    permutation: Dict[str, str],
    config: ExperimentConfig,
    model: HookedTransformer,
    eval_datasets: List[PermutedDataset],
    sgraph_dataset: SgraphDataset,
    batch_size: int,
    nb_replications: int,
) -> pd.DataFrame:
    results = []
    model.reset_hooks()
    patched_model = PatchedModel(
        model=model, sgraph_dataset=sgraph_dataset, communities={}
    )

    permutation_list = {
        k: [v] for k, v in permutation.items()
    }  # we need to convert the permutation to a list of list of features for compatibility with the hook_gen_targeted_rewrite function

    for i, steerers in enumerate(steerers_list):
        print(f"Running experiment {config.name} with parameter {config.param_list[i]}")

        hook_gen_tr = patched_model.hook_gen_targeted_rewrite(
            feature="querried_variable",
            list_of_components=steerers,
            feature_mapping=permutation_list,
        )

        # compute various stats about the set of steerers
        proportion_patched, nb_MLP, steerer_layers = compute_steerers_stats(
            steerers, model, config, i
        )

        # run the experiment
        for _ in range(nb_replications):
            tr_logits = patched_model.batched_patch(
                sgraph_dataset.tok_dataset, hook_gen_tr, batch_size=batch_size
            )
            print("Done with patching")  # TODO remove
            # evaluate the results on different datasets used as labels
            for eval_dataset in eval_datasets:
                print(f"Evaluating on {eval_dataset.name}")  # TODO remove
                result = evaluate_model(
                    model,
                    eval_dataset.permuted_dataset,
                    logits=tr_logits.cpu(),
                    end_position=WildPosition(
                        position=eval_dataset.orig_dataset.word_idx["END"], label="END"
                    ),
                )

                for variable in sgraph_dataset.feature_ids_to_names[
                    "querried_variable"
                ]:
                    for metric in ALL_METRICS:
                        results.append(
                            {
                                "Dataset": eval_dataset.name,
                                "Metric": metric,
                                "Querried variable": variable,
                                "Proportion patched": proportion_patched,
                                "Parameter": config.param_list[i],
                                "Result": result[f"{variable}_{metric}_mean"],
                                "Layers": steerer_layers,
                                "Number MLP": nb_MLP,
                            }
                        )
    return pd.DataFrame.from_records(results)



def run_cross_dataset_steering_experiment(
    steerers_list: List[List[ModelComponent]],
    config: ExperimentConfig,
    model: HookedTransformer,
    source_dataset: NanoQADataset,  # C1 Q1
    target_dataset: NanoQADataset,  # C2 Q2
    chimera_dataset: NanoQADataset,  # C2 Q1
    control_dataset: NanoQADataset,  # C3 Q3
    sgraph_dataset: SgraphDataset,
    batch_size: int,
    nb_replications: int,
):
    # just a simple batch patching of the steerers from orig_dataset to alt_dataset
    results = []

    # 1. activation store on the original dataset
    # cannot use the method `getPatchingHooksByIdx` because it's for patching _inside_ a given dataset
    #  I create my hooks my selfs from the cache
    # 2. I need to give the position to the patching hook -> OKAY

    cache = ActivationStore(
        listOfComponents=[], model=model, dataset=source_dataset.prompts_tok
    )

    for run in range(nb_replications):
        for i, steerers in enumerate(steerers_list):
            print(
                f"Running experiment {config.name} with parameter {config.param_list[i]}"
            )

            proportion_patched, nb_MLP, steerer_layers = compute_steerers_stats(
                steerers, model, config, i
            )

            cache.change_component_list(steerers)  # C1 Q1

            all_logits = []
            for b in range(0, len(source_dataset), batch_size):
                # generate the hooks for the batch
                patchingHooks = []

                # same source and target because we're patching accross dataset, in an aligned fashion
                source_idx = [
                    idx for idx in range(b, min(b + batch_size, len(source_dataset)))
                ]
                target_idx = [
                    idx for idx in range(b, min(b + batch_size, len(source_dataset)))
                ]
                for c in steerers:
                    patchingHooks.append(
                        (
                            c.hook_name,
                            partial(
                                component_patching_hook,
                                component=c,
                                cache=cache.transformerLensCache[c.hook_name][
                                    source_idx
                                ],
                                source_idx=source_idx,
                                target_idx=target_idx,
                                source_position=WildPosition(
                                    source_dataset.word_idx["END"], label="END SOURCE"
                                ),  # important to add the position for the source
                                verbose=False,
                            ),
                        )
                    )
                logits = model.run_with_hooks(
                    target_dataset.prompts_tok[target_idx],
                    fwd_hooks=patchingHooks,
                )
                all_logits.append(logits)
            all_logits = torch.cat(all_logits)

            for dataset in [
                source_dataset,
                target_dataset,
                chimera_dataset,
                control_dataset,
            ]:
                result = evaluate_model(
                    model,
                    dataset,
                    logits=all_logits.cpu(),
                    end_position=WildPosition(
                        position=target_dataset.word_idx["END"], label="END"
                    ),
                )

                for variable in sgraph_dataset.feature_ids_to_names[
                    "querried_variable"
                ]:
                    for metric in ALL_METRICS:
                        results.append(
                            {
                                "Dataset": dataset.name,
                                "Metric": metric,
                                "Querried variable": variable,
                                "Proportion patched": proportion_patched,
                                "Parameter": config.param_list[i],
                                "Result": result[f"{variable}_{metric}_mean"],
                                "Layers": steerer_layers,
                                "Number MLP": nb_MLP,
                            }
                        )
    return pd.DataFrame.from_records(results)


xp_name = "EleutherAI-pythia-2.8b-z-nanoQA-unruffled_mclean"  # EleutherAI-pythia-2.8b-z-nanoQA-unruffled_mclean gpt2-small-z-nanoQA-condescending_bassi
xp_path = "../xp"
model_name = None
exclude_mlp_zero = False
show_fig = True


IMPORTANCE_PERCENTILE = 0.7


def have_distinct_names(objects: Union[List[ExperimentConfig], List[PermutedDataset]]):
    return len(set(c.name for c in objects)) == len(objects)


# %%
def tr_nanoQA(
    xp_name: str,
    xp_path: str = "../xp",
    model_name: Optional[str] = None,
    stop_after_nm=False,
    show_plots=False,
):
    # %%
    # -2. Load the data of the experiment|
    # -1. define a list of experiment configurations
    # 0. For each config -> define a list of parameters for each technique. Many config can lead to the same technique
    # various numbers of random components
    # various layer of the residual stream
    # various threshold for the rand index
    # Given a set of steerers coming from 20%, various order of them
    # 1. create the list of senders.
    # A dict experimental param -> list of steerers
    # 2. Run the experiments
    # A dict of experiment config -> list of results
    # 3. plot the results
    # A single plot for each metric and question type.

    path, model_name, clean_model_name, dataset_name = load_config(
        xp_name, xp_path, model_name  # type: ignore
    )
    assert model_name is not None
    clean_model_name = model_name.replace("/", "-")

    path_to_plots = f"plots/{dataset_name}-tr/{xp_name}"
    if not os.path.exists(path_to_plots):
        os.mkdir(path_to_plots)

    assert clean_model_name in xp_name, "Model name should be in the xp name"

    # sgraph_dataset = load_object(path, "sgraph_dataset.pkl")
    # dataset = load_object(path, "dataset.pkl")
    comp_metric = load_object(path, "comp_metric.pkl").cpu().numpy()
    mean_comp_metric = comp_metric.mean(axis=-1)
    all_pnet_data = load_object(path, "all_data.pkl")

    model = HookedTransformer.from_pretrained(
        model_name, device="cuda"
    )  # raw model to hard reset the hooks

    # %% define dataset

    #  0.5 Define datasets

    QUERRIED_VARIABLES = [
        "character_name",
        "city",
        "character_occupation",
        "day_time",
        "season",
    ]

    nano_qa_dataset = NanoQADataset(
        nb_samples=100,
        nb_variable_values=5,
        tokenizer=model.tokenizer,  # type: ignore
        seed=43,
        querried_variables=QUERRIED_VARIABLES,
    )

    sgraph_dataset = SgraphDataset(
        feature_dict=get_nano_qa_features_dict(nano_qa_dataset),
        tok_dataset=nano_qa_dataset.prompts_tok,
        str_dataset=nano_qa_dataset.prompts_text,
    )

    #  check if the model is good enough TODO uncomment
    d = evaluate_model(model, nano_qa_dataset, batch_size=20)
    for querried_feature in nano_qa_dataset.querried_variables:  # type: ignore
        assert d[f"{querried_feature}_top1_mean"] > 0.5

    print_performance_table(d)

    print("End of the initialization")

    # %%

    # %%
    #  0. Define the list of experiment configurations

    configs = [
        ExperimentConfig(
            steerer_type="sgraph",
            param_list=list(np.linspace(1.0, 0.5, 5)),  # threshold for the rand index
            name="Swap graph component",
            nb_replications=1,
        ),
        ExperimentConfig(
            steerer_type="resid",
            param_list=list(
                np.linspace(-1, model.cfg.n_layers - 1, 5).astype(int)
            ),  # layer at which the residual stream is patched
            name="Residual stream",
            nb_replications=1,
        ),
        ExperimentConfig(
            steerer_type="random",
            param_list=list(np.linspace(0.0, 0.5, 5)),  # number of random components
            name="random component",
            nb_replications=1,
        ),
    ]
    assert have_distinct_names(configs), "Experiment configs should have distinct names"

    PERMUTATION = {
        "character_name": "city",
        "city": "day_time",
        "day_time": "season",
        "season": "character_occupation",
        "character_occupation": "character_name",
    }

    IDENTITY_PERMUTATION = {
        "city": "city",
        "character_name": "character_name",
        "character_occupation": "character_occupation",
        "day_time": "day_time",
        "season": "season",
    }

    PERMUTATION_CONTROL = {
        "city": "character_name",
        "character_name": "character_occupation",
        "character_occupation": "season",
        "season": "day_time",
        "day_time": "city",
    }

    permuted_dataset = PermutedDataset(
        permutatiom=PERMUTATION,
        orig_dataset=nano_qa_dataset,
        name="permuted dataset",
    )  # create a variable to be reused later

    eval_datasets = [
        PermutedDataset(
            orig_dataset=nano_qa_dataset,
            name="original dataset",
            permutatiom=IDENTITY_PERMUTATION,
        ),
        permuted_dataset,
        PermutedDataset(
            permutatiom=PERMUTATION_CONTROL,
            orig_dataset=nano_qa_dataset,
            name="control dataset",
        ),
    ]

    assert have_distinct_names(eval_datasets), "Datasets should have distinct names"

    # %%
    # l = model(sgraph_dataset.tok_dataset)
    # d = evaluate_model(
    #     model, eval_datasets[0].orig_dataset, logits=l, end_position=end_position
    # )
    # print_performance_table(d)  # TODO : find why this is buggy !!!
    # # %%

    # print(eval_datasets[0].orig_dataset.prompts_text[0])
    # print(eval_datasets[0].permuted_dataset.prompts_text[0])
    # %%

    #  1. Create the list of steerers
    # A dict experimental config -> list of steerers

    end_position = WildPosition(position=nano_qa_dataset.word_idx["END"], label="END")

    list_components = [
        compo_name_to_object(c, end_position, model.cfg.n_heads)
        for c in all_pnet_data.keys()
    ]
    all_mlp_components = [str(c) for c in list_components if c.name == "mlp"]
    steerers = {}
    for config in configs:
        steerers[config.name] = config_to_steerers(
            config,
            all_components=list_components,
            all_sgraph_data=all_pnet_data,
            comp_metric_array=mean_comp_metric,
            model=model,
            position=end_position,
            sent_feature="querried_variable",
        )

    # Print the length of the steerers
    for config in configs:
        print(f"{config.name} : {[len(x) for x in steerers[config.name]]}")
        print(
            f"{config.name} MLP: {[len([y for y in x if not y.is_head()]) for x in steerers[config.name]]}"
        )

    # %% 2. Run the experiments

    baseline_dataset = None
    RUN_INSIDE_DATASET = False
    if RUN_INSIDE_DATASET:
        results = []
        for config in configs[:1]:
            print(f"Running experiment {config.name}")
            df = run_steering_experiment(
                steerers_list=steerers[config.name],
                permutation=PERMUTATION,
                config=config,
                model=model,
                eval_datasets=eval_datasets,
                sgraph_dataset=sgraph_dataset,
                batch_size=10,
                nb_replications=config.nb_replications,
            )
            df["experiment_name"] = config.name
            results.append(df)
        results = pd.concat(results)
        baseline_dataset = permuted_dataset.permuted_dataset

    #  ALTERNATIVE EXPEIMENT C1 Q1, C2 Q2 -> Q1(C2) ?
    else:
        # 
        nano_qa_dataset_source = NanoQADataset(
            nb_samples=100,
            nb_variable_values=5,
            tokenizer=model.tokenizer,  # type: ignore
            seed=43,
            querried_variables=QUERRIED_VARIABLES,
            name="source dataset",
        )

        nano_qa_dataset_target = NanoQADataset(
            nb_samples=100,
            nb_variable_values=5,
            tokenizer=model.tokenizer,  # type: ignore
            seed=44,
            querried_variables=QUERRIED_VARIABLES,
            name="target dataset",
        )

        chimera_dataset = nano_qa_dataset_target.question_from(
            nano_qa_dataset_source, name="chimera dataset"
        )

        nano_qa_dataset_control = NanoQADataset(
            nb_samples=100,
            nb_variable_values=5,
            tokenizer=model.tokenizer,  # type: ignore
            seed=45,
            querried_variables=QUERRIED_VARIABLES,
            name="control dataset",
        )
        
        end_position = WildPosition(position=nano_qa_dataset_target.word_idx["END"], label="END")
        
        for c in list_components: # TODO factor this more cleanly
            c.position = end_position # update the position of the components
        steerers = {}
        for config in configs:
            steerers[config.name] = config_to_steerers(
                config,
                all_components=list_components,
                all_sgraph_data=all_pnet_data,
                comp_metric_array=mean_comp_metric,
                model=model,
                position=end_position,
                sent_feature="querried_variable",
            )
        # 

        results = []
        for config in configs:
            print(f"Running CROSS DATASET experiment {config.name}")
            df = run_cross_dataset_steering_experiment(
                steerers_list=steerers[config.name],
                config=config,
                model=model,
                source_dataset=nano_qa_dataset_source,
                target_dataset=nano_qa_dataset_target,
                chimera_dataset=chimera_dataset,
                control_dataset=nano_qa_dataset_control,
                sgraph_dataset=sgraph_dataset,
                batch_size=10,
                nb_replications=config.nb_replications,
            )
            df["experiment_name"] = config.name
            results.append(df)
        results = pd.concat(results)
        baseline_dataset = chimera_dataset

    # %%
    assert baseline_dataset is not None, "baseline dataset should not be None"
    baseline_results = evaluate_model(model, baseline_dataset)
    # %%

    random_guess = compute_random_guess(baseline_dataset)

    # %% 3. Plot the results
    df = results.copy()

    for metric in df["Metric"].unique():
        df_filtered = df[(df["Metric"] == metric)]

        fig = px.line(
            df_filtered,
            x="Proportion patched",
            y="Result",
            color="Dataset",
            line_group="experiment_name",
            line_dash="experiment_name",
            facet_row="Querried variable",
            labels={
                "Proportion patched": "Proportion Patched",
                "Result": "Result",
                "Dataset": "Dataset",
            },
            title="Result vs Proportion Patched for Different Datasets and Experiments <br> <b>(Metric: "
            + metric
            + ")</b>",
        )

        # add horizontal line for baseline with the legend "baseline". There should be one baseline line for
        # each querried variable, the value is the key f"{variable}_{metric}_mean"
        nb_q_var = len(df_filtered["Querried variable"].unique())
        for i, querried_variable in enumerate(
            df_filtered["Querried variable"].unique()
        ):
            baseline_value = baseline_results[f"{querried_variable}_{metric}_mean"]
            fig.add_hline(
                y=baseline_value,
                line_dash="dashdot",
                row=nb_q_var - i,
                col=1,
            )
            fig.add_hline(
                y=random_guess[f"{querried_variable}_{metric}_random_guess"],
                line_dash="dashdot",
                row=nb_q_var - i,
                col=1,
                line_color="grey",
            )

            print(querried_variable, baseline_value)

        fig.add_trace(  # add legend entry
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color="black", dash="dashdot"),
                name="Perf on permuted dataset",
            )
        )
        fig.add_trace(  # add legend entry
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color="grey", dash="dashdot"),
                name="Random guess",
            )
        )

        fig.update_layout(height=1300)
        fig.show()
        
        fig.write_html(path_to_plots+f"/{metric}.html")

    # px.scatter(x="Proportion patched", y="Result", color="Dataset", data_frame=results, facet_col="Querried variable", facet_row="Metric").show()


# %%
