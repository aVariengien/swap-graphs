# %%
import os

os.chdir("..")
# %%
from .questions import (
    QUESTIONS,
    QUESTION_PROMPT,
    gen_question_prompt,
)
from .nanostories import NANOSTORIES
from .narrative_variables import (
    QUERRIED_NARRATIVE_VARIABLES_PRETTY_NAMES,
)
from .nano_qa_utils import (
    check_tokenizer,
    print_performance_table,
)

from patching_networks.utils import wrap_str

from patching_networks.core import objects_to_unique_ids, objects_to_strings

import attrs
from typing import Dict, List, Optional, Callable
from jaxtyping import Float
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
import numpy as np
import torch
from transformer_lens import (
    ActivationCache,
    FactoredMatrix,
    HookedTransformer,
    HookedTransformerConfig,
)
import transformer_lens
from tqdm import tqdm
import time
from pprint import pprint

torch.set_grad_enabled(False)


# %%
@attrs.define
class NanoQADataset:
    """Dataset for QA on nanostories.
    querried_variables contains a list of the variables that are querried in the questions. If not set, all variables can be querried.
    K_shot defines the number of examples included in context. (currently not implem, only K_shot=0 is supported)
    """

    nb_samples: int = attrs.field()
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast = attrs.field()
    seed: int = attrs.field(default=42)
    K_shot: int = attrs.field(default=1)
    rng = attrs.field(init=False)
    nanostories: List[Dict] = attrs.field(init=False)
    questions: List[Dict[str, str]] = attrs.field(init=False)
    querried_variables: Optional[List[str]] = attrs.field(init=True, default=None)
    answer_texts: List[str] = attrs.field(init=False)
    prompts_text: List[str] = attrs.field(init=False)
    answer_tokens: List[int] = attrs.field(init=False)
    answer_first_token_texts: List[str] = attrs.field(init=False)
    word_idx: Dict[str, List[int]] = attrs.field(init=False)
    prompts_tok: torch.Tensor = attrs.field(init=False)

    def __attrs_post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        self.nanostories = list(
            self.rng.choice(np.array(NANOSTORIES), size=self.nb_samples)
        )

        if self.querried_variables is not None:
            for v in self.querried_variables:
                assert (
                    v in QUERRIED_NARRATIVE_VARIABLES_PRETTY_NAMES
                ), "Invalid querried variable."

            possible_questions = []
            for q in QUESTIONS:
                if q["querried_variable"] in self.querried_variables:
                    possible_questions.append(q)
        else:
            possible_questions = QUESTIONS

        self.questions = list(
            self.rng.choice(np.array(possible_questions), size=self.nb_samples)
        )

        assert check_tokenizer(
            self.tokenizer
        ), "Your tokenizer is such that there are collision in the first token of the answer."

        self.answer_texts = []
        self.answer_tokens = []
        self.prompts_text = []
        self.answer_first_token_texts = []
        self.word_idx = {"END": []}

        for i in range(self.nb_samples):
            querried_variable = self.questions[i]["querried_variable"]
            answer_str = (
                " " + self.nanostories[i]["seed"][querried_variable]
            )  ##we add the leading space
            answer_tokens = torch.tensor(self.tokenizer([answer_str])["input_ids"])[
                0
            ]  # possibly multiple tokens

            self.answer_tokens.append(int(answer_tokens[0]))
            self.answer_first_token_texts.append(
                self.tokenizer.decode([answer_tokens[0]])
            )
            self.answer_texts.append(answer_str)

            qa_prompt_text = gen_question_prompt(
                self.nanostories[i]["story"], self.questions[i]
            )
            self.prompts_text.append(qa_prompt_text)

            self.word_idx["END"].append(
                len(torch.tensor(self.tokenizer([qa_prompt_text])["input_ids"])[0]) - 1
            )
        self.prompts_tok = torch.tensor(
            self.tokenizer(self.prompts_text, padding=True)["input_ids"]
        )

    def __len__(self):
        return self.nb_samples


TokDataset = Float[torch.Tensor, "batch seq"]


def get_nano_qa_features_dict(
    nano_qa_dataset: NanoQADataset,
) -> Dict[str, List[str]]:
    querried_variable = objects_to_strings(
        [
            nano_qa_dataset.questions[i]["querried_variable"]
            for i in range(len(nano_qa_dataset))
        ]
    )

    questions = [
        nano_qa_dataset.questions[i]["question"] for i in range(len(nano_qa_dataset))
    ]

    answer_first_token = objects_to_strings(nano_qa_dataset.answer_first_token_texts)

    nb_token_answer = objects_to_strings(
        [
            len(
                torch.tensor(
                    nano_qa_dataset.tokenizer([nano_qa_dataset.answer_texts[i]])[
                        "input_ids"
                    ]
                )[0]
            )
            for i in range(len(nano_qa_dataset))
        ]
    )

    feature_dict = {
        "querried_variable": querried_variable,
        "answer_first_token": answer_first_token,
        "nb_token_answer": nb_token_answer,
        "questions": questions,
    }

    for narr_variable in nano_qa_dataset.nanostories[0][
        "seed"
    ].keys():  # add every variable in the seed to the feature dict
        values = [
            nano_qa_dataset.nanostories[i]["seed"][narr_variable]
            for i in range(len(nano_qa_dataset))
        ]

        if narr_variable == "master_story":
            values, _ = objects_to_unique_ids(values)

        feature_dict[narr_variable] = objects_to_strings(values)
    return feature_dict


def evaluate_model(
    model: HookedTransformer,
    nano_qa_dataset: NanoQADataset,
    batch_size=10,
    all=False,
    print=False,
):
    logits = torch.zeros(
        (nano_qa_dataset.nb_samples, model.cfg.n_ctx, model.cfg.d_vocab_out)
    )

    for i in tqdm(range(0, nano_qa_dataset.nb_samples, batch_size)):
        batch = nano_qa_dataset.prompts_text[i : i + batch_size]
        batch_logits = model(batch, prepend_bos=False)
        logits[i : i + len(batch), : batch_logits.shape[1]] = model(
            batch, prepend_bos=False
        )
    end_logits = logits[
        torch.arange(nano_qa_dataset.nb_samples), nano_qa_dataset.word_idx["END"]
    ]

    end_probs = torch.nn.functional.softmax(end_logits, dim=-1)

    questions_types = list(QUERRIED_NARRATIVE_VARIABLES_PRETTY_NAMES.keys())
    perf_per_variable = {questions_types[i]: {} for i in range(len(questions_types))}

    for question_type in perf_per_variable:
        perf_per_variable[question_type]["prob"] = []
        perf_per_variable[question_type]["rank"] = []
        perf_per_variable[question_type]["top1"] = []

    for i in range(nano_qa_dataset.nb_samples):
        question_type = nano_qa_dataset.questions[i]["querried_variable"]
        perf_per_variable[question_type]["prob"].append(
            end_probs[i][nano_qa_dataset.answer_tokens[i]].item()
        )

        rank = (
            torch.where(
                torch.argsort(end_probs[i], descending=True)
                == nano_qa_dataset.answer_tokens[i]
            )[0][0]
            + 1
        )

        top1 = int(
            torch.argsort(end_probs[i], descending=True)[0]
            == nano_qa_dataset.answer_tokens[i]
        )

        perf_per_variable[question_type]["rank"].append(rank)

        perf_per_variable[question_type]["top1"].append(top1)

    summary_perf_per_variable = {}
    for question_type in perf_per_variable:
        for metric in perf_per_variable[question_type]:
            summary_perf_per_variable[question_type + "_" + metric + "_mean"] = np.mean(
                perf_per_variable[question_type][metric]
            )
            summary_perf_per_variable[question_type + "_" + metric + "_std"] = np.std(
                perf_per_variable[question_type][metric]
            )

            summary_perf_per_variable[
                question_type + "_" + metric + "_all"
            ] = perf_per_variable[question_type][metric]

    return summary_perf_per_variable


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "EleutherAI/pythia-70m-deduped",
        revision="step3000",
    )
    tokenizer.pad_token = tokenizer.eos_token

    nano_qa_dataset = NanoQADataset(nb_samples=100, tokenizer=tokenizer, seed=43)

    f_dict = get_nano_qa_features_dict(nano_qa_dataset)
    # %%
    t1 = time.time()
    d = evaluate_model(model, nano_qa_dataset, batch_size=10)  # type: ignore
    print_performance_table(d)
    t2 = time.time()
    print(t2 - t1)
    # %%
    model_name = "EleutherAI/Pythia-2.8b"  # "EleutherAI/gpt-neo-2.7B"  #

    model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")

    # %% manual inspection
    for i in range(10):
        print("===========")
        print(wrap_str(nano_qa_dataset.prompts_text[i]))
        print()

        transformer_lens.utils.test_prompt(
            prompt=nano_qa_dataset.prompts_text[i],
            answer=nano_qa_dataset.answer_first_token_texts[i],
            model=model,
            prepend_space_to_answer=False,
            print_details=True,
            prepend_bos=False,
            top_k=10,
        )

    # %%
    evaluate_model(model, nano_qa_dataset, batch_size=100)
    # %%
