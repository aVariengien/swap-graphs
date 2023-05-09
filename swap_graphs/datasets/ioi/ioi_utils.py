# %%
from .ioi_dataset import IOIDataset, OBJECTS, PLACES, NAMES_GENDER
import torch
from jaxtyping import Float, Int
from typing import Callable, List, Union, Optional, Tuple, Dict, Any, Sequence
from swap_graphs.core import WildPosition, objects_to_strings


def handle_all_and_std(returning, all, std):
    """
    For use by the below functions. Lots of options!!!
    """

    if all and not std:
        return returning
    if std:
        if all:
            first_bit = (returning).detach().cpu()
        else:
            first_bit = (returning).mean().detach().cpu()
        return first_bit, torch.std(returning).detach().cpu()
    return (returning).mean().detach().cpu()


def logit_diff(
    model,
    ioi_dataset: IOIDataset,
    all=False,
    std=False,
    both=False,
):  # changed by Arthur to take dataset object, :pray: no big backwards compatibility issues
    """
    Difference between the IO and the S logits at the "to" token
    """

    logits = model(ioi_dataset.prompts_tok.long()).detach()

    IO_logits = logits[
        torch.arange(len(ioi_dataset)),
        ioi_dataset.word_idx["END"],
        ioi_dataset.io_tokenIDs,
    ]
    S_logits = logits[
        torch.arange(len(ioi_dataset)),
        ioi_dataset.word_idx["END"],
        ioi_dataset.s_tokenIDs,
    ]

    if both:
        return handle_all_and_std(IO_logits, all, std), handle_all_and_std(
            S_logits, all, std
        )

    else:
        return handle_all_and_std(IO_logits - S_logits, all, std)


def probs(
    model, ioi_dataset: IOIDataset, all=False, std=False, type="io", verbose=False
):
    """
    IO probs
    """
    logits = model(
        ioi_dataset.prompts_tok.long()
    ).detach()  # batch * sequence length * vocab_size
    end_logits = logits[
        torch.arange(len(ioi_dataset)), ioi_dataset.word_idx["END"], :
    ]  # batch * vocab_size

    end_probs = torch.softmax(end_logits, dim=1)

    if type == "io":
        token_ids = ioi_dataset.io_tokenIDs
    elif type == "s":
        token_ids = ioi_dataset.s_tokenIDs
    elif type == "io2":
        token_ids = ioi_dataset.io2_tokenIDs
    else:
        raise ValueError("type must be io or s")

    assert len(end_probs.shape) == 2
    io_probs = end_probs[torch.arange(ioi_dataset.N), token_ids]
    if verbose:
        print(io_probs)
    return handle_all_and_std(io_probs, all, std)


def assert_model_perf_ioi(model, ioi_dataset):
    ld = logit_diff(model, ioi_dataset)
    io_prob = probs(model, ioi_dataset, type="io")

    assert (
        ld.item() > 2.5 and io_prob.item() > 0.15  # type: ignore
    ), f"The model is not good enough on the dataset (ld={ld}, io_prob = {io_prob}). There might be a setup problem."

    print(f"Model loading test passed. (ld={ld}, io_prob = {io_prob})")
    return True


def logit_diff_comp(
    logits_target: Float[torch.Tensor, "batch seq vocab"],
    logits_source: Float[torch.Tensor, "batch seq vocab"],
    target_seqs: torch.Tensor,
    target_idx: List[int],
    ioi_dataset: IOIDataset,
    keep_sign: bool = False,
):
    position_to_evaluate = WildPosition(ioi_dataset.word_idx["END"], label="END")

    if keep_sign:
        comp_fn = torch.mean
    else:
        comp_fn = torch.norm

    # print(
    #     len(target_seqs),
    #     len(position_to_evaluate.positions_from_idx(target_idx)),
    #     len(ioi_dataset.io_tokenIDs[target_idx]),
    # )

    IO_logits_target = logits_target[
        torch.arange(len(target_seqs)),
        position_to_evaluate.positions_from_idx(target_idx),
        ioi_dataset.io_tokenIDs[target_idx],
    ]

    S_logits_target = logits_target[
        torch.arange(len(target_seqs)),
        position_to_evaluate.positions_from_idx(target_idx),
        ioi_dataset.s_tokenIDs[target_idx],
    ]

    IO_logits_source = logits_source[
        torch.arange(len(target_seqs)),
        position_to_evaluate.positions_from_idx(target_idx),
        ioi_dataset.io_tokenIDs[target_idx],
    ]

    S_logits_source = logits_source[
        torch.arange(len(target_seqs)),
        position_to_evaluate.positions_from_idx(target_idx),
        ioi_dataset.s_tokenIDs[target_idx],
    ]

    if ioi_dataset.wild_template:
        IO2_logits_source = logits_source[
            torch.arange(len(target_seqs)),
            position_to_evaluate.positions_from_idx(target_idx),
            ioi_dataset.io2_tokenIDs[target_idx],
        ]

        IO2_logits_target = logits_target[
            torch.arange(len(target_seqs)),
            position_to_evaluate.positions_from_idx(target_idx),
            ioi_dataset.io2_tokenIDs[target_idx],
        ]

        diff_in_logit_diff = comp_fn(
            torch.cat(
                [
                    (
                        (IO_logits_target - S_logits_target)
                        - (IO_logits_source - S_logits_source)
                    ).unsqueeze(1),
                    (
                        (IO2_logits_target - S_logits_target)
                        - (IO2_logits_source - S_logits_source)
                    ).unsqueeze(1),
                ],
                dim=1,
            ),
            dim=1,
        )
    else:
        diff_in_logit_diff = comp_fn(
            (
                (IO_logits_target - S_logits_target)
                - (IO_logits_source - S_logits_source)
            ).unsqueeze(1),
            dim=1,
        )

    return diff_in_logit_diff


order_to_number = {"BAB": 0, "ABB": 1, "S1IO1IO2": 0, "IO1S1IO2": 1, "IO1IO2S1": 2}


TokDataset = Float[torch.Tensor, "batch seq"]
FeatureFct = Callable[[TokDataset], List[int]]


def get_ioi_features_dict(ioi_dataset: IOIDataset) -> Dict[str, List[str]]:
    S1_token = [ioi_dataset.prompts_metadata[i]["S"] for i in range(len(ioi_dataset))]
    IO_token = [ioi_dataset.prompts_metadata[i]["IO"] for i in range(len(ioi_dataset))]
    position = [
        ioi_dataset.prompts_metadata[i]["order"] for i in range(len(ioi_dataset))
    ]
    effective_obj = OBJECTS + ["none"]
    obj = [ioi_dataset.prompts_metadata[i]["[OBJECT]"] for i in range(len(ioi_dataset))]
    if not "none" in [ioi_dataset.prompts_metadata[i] for i in range(len(ioi_dataset))]:
        for i in range(len(ioi_dataset)):
            if "said to" in ioi_dataset.prompts_text[i]:
                obj[i] = "none"
    obj = objects_to_strings(obj)
    place = [
        ioi_dataset.prompts_metadata[i]["[PLACE]"] for i in range(len(ioi_dataset))
    ]

    if ioi_dataset.prompt_type == "mixed":
        modulo = len(ioi_dataset.templates) // 2
    else:
        modulo = len(ioi_dataset.templates)
    template_idx = objects_to_strings(
        [
            ioi_dataset.prompts_metadata[i]["TEMPLATE_IDX"] % modulo
            for i in range(len(ioi_dataset))
        ]
    )
    sentence_types = []
    for i in range(len(ioi_dataset)):
        if "said to" in ioi_dataset.prompts_text[i]:
            sentence_types.append("said_to")
        elif (
            "give it" in ioi_dataset.prompts_text[i]
            or "gave it" in ioi_dataset.prompts_text[i]
        ):
            sentence_types.append("give_it")
        else:
            sentence_types.append("other")
    S_gender = objects_to_strings(
        [
            NAMES_GENDER[ioi_dataset.prompts_metadata[i]["S"]]
            for i in range(len(ioi_dataset))
        ]
    )
    IO_gender = objects_to_strings(
        [
            NAMES_GENDER[ioi_dataset.prompts_metadata[i]["IO"]]
            for i in range(len(ioi_dataset))
        ]
    )

    return {
        "S1 token": S1_token,
        "IO token": IO_token,
        "Order of first names": position,
        "Object": obj,
        "Place": place,
        "Template index": template_idx,
        "IO gender": IO_gender,
        "S gender": S_gender,
        "Sentence type": sentence_types,
    }


def d(
    f: Callable[[Float[torch.Tensor, "pos"]], int],
):
    def g(dataset: Float[torch.Tensor, "batch pos"]):
        return [f(v) for v in dataset]

    return g


def uniform_color(dataset: Float[torch.Tensor, "batch pos"]):
    return [0 for v in dataset]


# %%
