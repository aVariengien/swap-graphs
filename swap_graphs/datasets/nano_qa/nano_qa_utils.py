# %%

from .narrative_variables import (
    NARRATIVE_VARIABLES,
    QUERRIED_NARRATIVE_VARIABLES_PRETTY_NAMES,
)
import torch
import numpy as np

# %%


def check_tokenizer_nanoQA(tokenizer):
    safe = True
    existing_first_tokens = set()
    for variable_name, variable_values in NARRATIVE_VARIABLES.items():
        if variable_name in QUERRIED_NARRATIVE_VARIABLES_PRETTY_NAMES:
            for x in variable_values:
                first_token = torch.tensor(tokenizer([" " + x])["input_ids"])[0][0]
                if first_token in existing_first_tokens:
                    print(x, first_token)
                    safe = False
                existing_first_tokens.add(first_token)
    return safe


# %%
def print_performance_table(d):
    header = (
        "Feature             | Mean (Prob)  | Mean (Rank)  | Top1 Accuracy         "
    )
    divider = "-" * len(header)
    print(header)
    print(divider)

    features = {
        "character_name": "Character Name",
        "character_occupation": "Character Occupation",
        "city": "City",
        "day_time": "Day Time",
        "season": "Season",
    }

    for feature, feature_label in features.items():
        prob_mean = d[f"{feature}_prob_mean"]
        prob_std = d[f"{feature}_prob_std"]
        rank_mean = d.get(f"{feature}_rank_mean", None)
        rank_std = d.get(f"{feature}_rank_std", None)
        top1_mean = d[f"{feature}_top1_mean"]
        top1_std = d[f"{feature}_top1_std"]
        if rank_mean is None or rank_std is None:
            rank_mean_str = "N/A"
        else:
            rank_mean_str = f"{rank_mean:.2f} ± {rank_std:.2f}"

        print(
            f"{feature_label:<20}| {prob_mean:.2f} ± {prob_std:.2f} | {rank_mean_str:<15}| {top1_mean:.2f} ± {top1_std:.2f}"
        )


# %%
