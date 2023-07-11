import transformer_lens.utils as utils
import plotly.express as px

NO_CIRCUITVIS = False
try:
    import circuitsvis  # circuitvis requires pytorch 1.8.1. It's not always installed.
except ImportError as e:
    NO_CIRCUITVIS = True
    pass  # module doesn't exist, deal with it.

# import circuitsvis as cv
import torch

import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Float, Int
from typing import Callable, List, Union, Optional, Tuple, Dict, Any, Sequence

from swap_graphs.core import WildPosition, ModelComponent, NOT_A_HEAD
import os
import pickle

from transformers import PreTrainedTokenizer
import random
import gc
import time


def objects_to_unique_ids(l: list):
    values = list(set([str(x) for x in l]))
    values.sort()
    return [values.index(str(x)) for x in l]


T = time.time()


def print_time(s: str):
    global T
    print(f"{s} in {time.time() - T:.2f} s")
    T = time.time()


def compo_name_to_type(comp_name: str):
    for l in ["q", "k", "v", "z"]:
        if "hook_" + l in comp_name:
            return l
    for l in ["resid_pre", "resid_post", "mlp", "attn"]:
        if l in comp_name:
            return l


def component_name_to_idx(name: str, n_heads) -> Tuple[int, int]:
    L = 99999
    H = 99999
    if "mlp" in name:
        H = n_heads
    else:
        h = name.split(".")[-1]
        H = int(h.split("@")[0].replace("h", ""))
        assert H < n_heads
    L = int(name.split(".")[1])

    return L, H


def compo_name_to_object(comp_name: str, position: WildPosition, nb_heads: int):
    l, h = component_name_to_idx(comp_name, nb_heads)
    if h == nb_heads:
        h = NOT_A_HEAD
    return ModelComponent(
        position=position, layer=l, head=h, name=compo_name_to_type(comp_name)
    )


def create_random_communities(
    list_compos: List[ModelComponent], n_samples: int, n_classes=5
) -> Dict[ModelComponent, Dict[int, int]]:
    random_commus = {}
    for c in list_compos:
        rd_commu = {}
        for k in range(n_samples):
            rd_commu[k] = random.randint(0, n_classes - 1)
        random_commus[c] = rd_commu.copy()
    return random_commus


def get_top_k_predictions(
    tokenizer: PreTrainedTokenizer, logits: torch.Tensor, k: int, idx: List[int]
) -> None:
    # Convert logits tensor to numpy array
    logits = logits[range(len(idx)), idx].cpu().detach()

    # Get the probabilities of the top-k predictions
    probs = torch.softmax(logits, dim=-1)
    all_top_k_probs, all_top_k_indices = torch.topk(probs, k, dim=-1)

    for j in range(len(idx)):
        top_k_probs = all_top_k_probs.detach().numpy()[j]
        top_k_indices = all_top_k_indices.detach().numpy()[j]

        # Get the top-k token predictions
        print(top_k_indices)
        top_k_tokens = [tokenizer.convert_ids_to_tokens([i])[0] for i in top_k_indices]

        # Print the top-k predictions and their probabilities
        print("Top-{} Predictions:".format(k))
        for i in range(k):
            print("{:<20} ({:.4f})".format(top_k_tokens[i], top_k_probs[i]))


def show_mtx(
    mtx,
    title="NO TITLE :(",
    color_map_label="Logit diff variation",
    nb_heads=12,
    save_path=None,
    display=True,
    height=600,
):
    """Show a plotly matrix with a centered color map. Designed to display results of path patching experiments."""
    max_val = float(max(abs(mtx.min()), abs(mtx.max())))
    x_labels = [f"h{i}" for i in range(nb_heads)] + ["mlp"]
    fig = px.imshow(
        mtx,
        title=title,
        labels={"x": "Head", "y": "Layer", "color": color_map_label},
        color_continuous_scale="RdBu",
        range_color=(-max_val, max_val),
        x=x_labels,
        y=[str(i) for i in range(mtx.shape[0])],
        aspect="equal",
        height=height,
    )
    # save the fig as a html file
    if save_path is not None:
        fig.write_html(f"{save_path}/{title}.html")

    if display:
        fig.show()
    else:
        return fig


def print_gpu_mem(step_name=""):
    print(
        f"{step_name} ~ {np.round(torch.cuda.memory_allocated()/1e9, 2)} Go allocated on GPU."
    )


def clean_gpu_mem():
    gc.collect()
    torch.cuda.empty_cache()


def imshow(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.imshow(
        utils.to_numpy(tensor),
        color_continuous_midpoint=0.0,
        color_continuous_scale="RdBu",
        labels={"x": xaxis, "y": yaxis},
        **kwargs,
    ).show(renderer)


def line(tensor, renderer=None, xaxis="", yaxis="", **kwargs):
    px.line(utils.to_numpy(tensor), labels={"x": xaxis, "y": yaxis}, **kwargs).show(
        renderer
    )


def scatter(x, y, xaxis="", yaxis="", caxis="", renderer=None, **kwargs):
    x = utils.to_numpy(x)
    y = utils.to_numpy(y)
    px.scatter(
        y=y, x=x, labels={"x": xaxis, "y": yaxis, "color": caxis}, **kwargs
    ).show(renderer)


def show_attn(model, text, layer):
    if isinstance(text, str):
        tokens = model.to_tokens(text, prepend_bos=False)
        gpt2_str_tokens = model.to_str_tokens(text, prepend_bos=False)
    elif isinstance(text, torch.Tensor):
        assert text.ndim == 1
        tokens = text
        gpt2_str_tokens = [str(t) for t in tokens.tolist()]
    else:
        raise ValueError("text must be a string or a tensor of tokens")

    print(tokens.device)
    gpt2_logits, gpt2_cache = model.run_with_cache(tokens, remove_batch_dim=True)
    print(type(gpt2_cache))

    try:
        print("yo")
        attention_pattern = gpt2_cache["pattern", layer, "attn"]
    except KeyError:
        attention_pattern = gpt2_cache["attn_scores", layer, "attn"]
        attention_pattern = torch.softmax(attention_pattern, dim=-1)
        print(attention_pattern.shape)

    print(f"Layer {layer} Head Attention Patterns:")
    if NO_CIRCUITVIS:
        print("CircuitVis not installed, cannot display attention patterns.")
        return

    return cv.attention.attention_patterns(  # type: ignore
        tokens=gpt2_str_tokens, attention=attention_pattern
    )


def plotHistLogLog(x, metric_name="NONE", only_y_log=False):
    min_dist = min(x)
    max_dist = max(x)
    # histogram on log scale.
    # Use non-equal bin sizes, such that they look equal on log scale.
    logbins = np.logspace(np.log10(min_dist + 1e-6), np.log10(max_dist), 100)
    bbins = logbins
    if only_y_log:
        plt.hist(x, bins=100, log=True)
        plt.xlabel(f"{metric_name}")
    else:
        plt.hist(x, bins=logbins, log=True)
        plt.xscale("log")
        plt.xlabel(f"{metric_name} (log)")
    plt.ylabel("Log # of samples")

    plt.show()


## distance metrics


def L2_dist(
    logits_target: Float[torch.Tensor, "batch seq vocab"],
    logits_source: Float[torch.Tensor, "batch seq vocab"],
    target_seqs: torch.Tensor,
    position_to_evaluate: int,
):
    probs_target = torch.nn.functional.softmax(
        logits_target[:, position_to_evaluate, :], dim=-1  # the original outputs
    )

    probs_source = torch.nn.functional.softmax(  # the patched outputs
        logits_source[:, position_to_evaluate, :], dim=-1
    )

    return torch.norm(probs_target - probs_source, dim=-1)


def L2_dist_in_context(
    logits_target: Float[torch.Tensor, "batch seq vocab"],
    logits_source: Float[torch.Tensor, "batch seq vocab"],
    target_seqs: torch.Tensor,
    position_to_evaluate: int,
):
    probs_target = torch.nn.functional.softmax(
        logits_target[:, position_to_evaluate, :], dim=-1  # the original outputs
    )

    probs_source = torch.nn.functional.softmax(  # the patched outputs
        logits_source[:, position_to_evaluate, :], dim=-1
    )

    norms = torch.zeros(len(target_seqs))
    for i in range(len(target_seqs)):
        in_ctx_token = target_seqs[i].unique()
        norms[i] = torch.norm(
            probs_target[i, in_ctx_token] - probs_source[i, in_ctx_token]
        )
    return norms * 100


def KL_div_sim(
    logits_target: Float[torch.Tensor, "batch seq vocab"],
    logits_source: Float[torch.Tensor, "batch seq vocab"],
    target_seqs: torch.Tensor,
    position_to_evaluate: Union[int, torch.Tensor, WildPosition],
    target_idx: List[int],
):
    if not (isinstance(position_to_evaluate, int)):
        assert (
            target_idx is not None
        ), "You should provide target_idx when position_to_evaluate is a tensor"

    if not isinstance(position_to_evaluate, WildPosition):
        position_to_evaluate = WildPosition(
            position_to_evaluate, label="position_to_evaluate"
        )

    # kl_div = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)
    log_probs_target = torch.nn.functional.log_softmax(
        logits_target[
            range(len(target_seqs)),
            position_to_evaluate.positions_from_idx(target_idx),
            :,
        ],
        dim=-1,  # log_softmax
    )

    log_probs_source = torch.nn.functional.log_softmax(  # log_softmax
        logits_source[
            range(len(target_seqs)),
            position_to_evaluate.positions_from_idx(target_idx),
            :,
        ],
        dim=-1,
    )
    return (torch.exp(log_probs_source) * (log_probs_source - log_probs_target)).sum(
        dim=-1
    )


def get_top_k_probs(logits, k):
    probs = torch.nn.functional.softmax(logits[:], dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, k)
    return top_k_probs, top_k_indices


def save_object(obj, path, name):
    with open(os.path.join(path, name), "wb") as f:
        pickle.dump(obj, f)


def load_object(path, name):
    with open(os.path.join(path, name), "rb") as f:
        return pickle.load(f)


def get_components_at_position(
    position: WildPosition,
    nb_layers: int,
    nb_heads: int,
    head_subpart: str = "z",
    include_mlp: bool = True,
) -> List[ModelComponent]:
    """Return the list of model components corresponding to the MLP blocks and attention heads at a given position."""
    components_to_search = []
    for l in range(nb_layers):
        for h in range(nb_heads):
            components_to_search.append(
                ModelComponent(
                    position=position,
                    layer=l,
                    name=head_subpart,
                    head=h,
                )
            )
        if include_mlp:
            components_to_search.append(
                ModelComponent(
                    position=position,
                    layer=l,
                    name="mlp",
                )
            )
    return components_to_search


def wrap_str(s: str, max_line_len=100):
    """Add skip line every max_line_len characters. Ensure that no word is cut in the middle."""
    words = s.split(" ")
    wrapped_str = ""
    line_len = 0
    for word in words:
        if "\n" in word:
            line_len = 0
        if line_len + len(word) > max_line_len:
            wrapped_str += "\n"
            line_len = 0
        wrapped_str += word + " "
        line_len += len(word) + 1
    return wrapped_str

def printw(s: str, max_line_len=100):
    print(wrap_str(s, max_line_len))


# %%
def load_config(xp_name: str, xp_path: str, model_name: Optional[str]):
    path = os.path.join(xp_path, xp_name)
    if os.path.exists(os.path.join(path, "config.pkl")):
        print("Loading config.pkl")
        config = load_object(path, "config.pkl")
        assert (
            config["xp_name"] == xp_name
        ), "xp_name in config.pkl does not match the xp_name argument."
        model_name = config["model_name"]
        dataset_name = config["dataset_name"]
    else:
        if "IOI" in xp_name:
            dataset_name = "IOI"
        elif "nanoQA" in xp_name:
            dataset_name = "nanoQA"
        else:
            raise ValueError("dataset_name must be specified")

    assert isinstance(
        model_name, str
    ), f"model_name must be a string {model_name}. You most likely forgot to specify it and not config was found in the experiment folder."
    assert isinstance(
        dataset_name, str
    ), f"dataset_name must be a string {dataset_name}"
    assert dataset_name in xp_name, f"{xp_name} does not contain {dataset_name}"

    MODEL_NAME = model_name.replace("/", "-")
    assert isinstance(MODEL_NAME, str)
    assert (
        MODEL_NAME in xp_name
    ), f"{xp_name} does not contain {MODEL_NAME}. Check your experiment name."

    return path, model_name, MODEL_NAME, dataset_name
