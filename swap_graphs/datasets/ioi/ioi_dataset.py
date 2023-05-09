# %%

import torch
from copy import deepcopy
from transformers import AutoTokenizer
from typing import Union, Literal, Any, NamedTuple
import random

ABBA_TEMPLATES = [
    #     "<|endoftext|>Then, {IO} and {S1} went to the {PLACE}. {S2} gave a {OBJECT} to",
    #     "<|endoftext|>On Friday, {IO} had a great day. Then {S1} met them. They all went to the {PLACE}. {S2} gave a {OBJECT} to",
    #     "<|endoftext|>Then, {IO} had a great day. Then, {S1} their teacher, met them. They all went to the {PLACE}. {S2} gave a {OBJECT} to",
    #     "<|endoftext|>It was told that {IO} was a funny character. The doctor {S1} met them. The three of them went to the {PLACE}. {S2} gave a {OBJECT} to",
    #     "<|endoftext|>People were told that {IO} was a funny character. {S1} was their friend for a long time. The three of them went to the {PLACE}. {S2} gave a {OBJECT} to",
    #     "<|endoftext|>After a week spend together, {IO} and {S1} felt they have known each other for several years. The three of them went to the {PLACE}. {S2} gave a {OBJECT} to",
    #     "<|endoftext|>After graduation, {IO} has been living alone for a long time. The doctor {S1} met the lonely character. Together, they decided to go to the {PLACE}. {S2} gave a {OBJECT} to",
    # ]
    "<|endoftext|>Then, {IO} and {S1} went to the {PLACE}. {S2} gave a {OBJECT} to",
    "<|endoftext|>Then, {IO} and {S1} had a lot of fun at the {PLACE}. {S2} gave a {OBJECT} to",
    "<|endoftext|>Then, {IO} and {S1} were working at the {PLACE}. {S2} decided to give a {OBJECT} to",
    "<|endoftext|>Then, {IO} and {S1} were thinking about going to the {PLACE}. {S2} wanted to give a {OBJECT} to",
    "<|endoftext|>Then, {IO} and {S1} had a long argument, and afterwards {S2} said to",
    "<|endoftext|>After {IO} and {S1} went to the {PLACE}, {S2} gave a {OBJECT} to",
    "<|endoftext|>When {IO} and {S1} got a {OBJECT} at the {PLACE}, {S2} decided to give it to",
    "<|endoftext|>When {IO} and {S1} got a {OBJECT} at the {PLACE}, {S2} decided to give the {OBJECT} to",
    "<|endoftext|>While {IO} and {S1} were working at the {PLACE}, {S2} gave a {OBJECT} to",
    "<|endoftext|>While {IO} and {S1} were commuting to the {PLACE}, {S2} gave a {OBJECT} to",
    "<|endoftext|>After the lunch, {IO} and {S1} went to the {PLACE}. {S2} gave a {OBJECT} to",
    "<|endoftext|>Afterwards, {IO} and {S1} went to the {PLACE}. {S2} gave a {OBJECT} to",
    "<|endoftext|>Then, {IO} and {S1} had a long argument. Afterwards {S2} said to",
    "<|endoftext|>The {PLACE} {IO} and {S1} went to had a {OBJECT}. {S2} gave it to",
    "<|endoftext|>Friends {IO} and {S1} found a {OBJECT} at the {PLACE}. {S2} gave it to",
]

# wild but two
#     "<|endoftext|>Then, {IO} and {S1} went to the {PLACE}. {S2} gave a {OBJECT} to",
#     "<|endoftext|>On Friday, {IO} had a great day. Then {S1} met them. They all went to the {PLACE}. {S2} gave a {OBJECT} to",
#     "<|endoftext|>Then, {IO} had a great day. Then, {S1} their teacher, met them. They all went to the {PLACE}. {S2} gave a {OBJECT} to",
#     "<|endoftext|>It was told that {IO} was a funny character. The doctor {S1} met them. The three of them went to the {PLACE}. {S2} gave a {OBJECT} to",
#     "<|endoftext|>People were told that {IO} was a funny character. {S1} was their friend for a long time. The three of them went to the {PLACE}. {S2} gave a {OBJECT} to",
#     "<|endoftext|>After a week spend together, {IO} and {S1} felt they have known each other for several years. The three of them went to the {PLACE}. {S2} gave a {OBJECT} to",
#     "<|endoftext|>After graduation, {IO} has been living alone for a long time. The doctor {S1} met the lonely character. Together, they decided to go to the {PLACE}. {S2} gave a {OBJECT} to",
# ]

BABA_TEMPLATES = []


def swap_substrings(s, substring_a, substring_b):
    """Swap two substrings in a string"""
    return (
        s.replace(substring_a, "___")
        .replace(substring_b, substring_a)
        .replace("___", substring_b)
    )


for template in ABBA_TEMPLATES:
    BABA_TEMPLATES.append(swap_substrings(template, "{IO}", "{S1}"))

# %%
WILD_TEMPLATES = [
    "People were told that {N1} was a funny character. {N2} was their friend for a long time while {N3} just arrived in town. The three of them went to the {PLACE}. {S2} gave a {OBJECT} to",
    "Then, {N1}, {N2} and {N3} went to the {PLACE}. {S2} gave a {OBJECT} to",
    "Then {N1} and {N2} and {N3} went to the {PLACE}. {S2} gave a {OBJECT} to",
    "Then, {N1} and a friend {N2} had a great day. Then, {N3} their teacher, met them. They all went to the {PLACE}. {S2} gave a {OBJECT} to",
    "It was told that {N1} was a funny character. The two friends {N2} and {N3} met them. The three of them went to the {PLACE}. {S2} gave a {OBJECT} to",
    "After a week spend together, {N1} and {N2} felt they have known each other for several years. They then met a new friend {N3}. The three of them went to the {PLACE}. {S2} gave a {OBJECT} to",
    "After graduation, {N1} has been living alone for a long time. The couple {N2} and {N3} met the lonely character. Together, they decided to go to the {PLACE}. {S2} gave a {OBJECT} to",
]

# WILD_TEMPLATES = [
#     "Then, {N1}, {N2} and {N3} went to the {PLACE}. {S2} gave a {OBJECT} to",
#     "Then, {N1}, {N2} and {N3} went to the {PLACE}. {S2} gave a {OBJECT} to",
#     "Then, {N1}, {N2} and {N3} had a lot of fun at the {PLACE}. {S2} gave a {OBJECT} to",
#     "Then, {N1}, {N2} and {N3} were working at the {PLACE}. {S2} decided to give a {OBJECT} to",
#     "Then, {N1}, {N2} and {N3} were thinking about going to the {PLACE}. {S2} wanted to give a {OBJECT} to",
#     "Then, {N1}, {N2} and {N3} had a long argument, and afterwards {S2} said to",
#     "After {N1}, {N2} and {N3} went to the {PLACE}, {S2} gave a {OBJECT} to",
#     "When {N1}, {N2} and {N3} got a {OBJECT} at the {PLACE}, {S2} decided to give it to",
# ]


PREFIXES = [
    # "It was a beautiful day.",
    # "It was the middle of fall and the day was sunny.",
    # "It was a rainy day. A storm had just passed. The streets were wet.",
    # "It was a cold day.",
    "",
]

WILD_TEMPLATES_PREFIXED = []
for prefix in PREFIXES:
    for template in WILD_TEMPLATES:
        WILD_TEMPLATES_PREFIXED.append("<|endoftext|>" + prefix + " " + template)

ALL_WILD_TEMPLATES = []

if "{S1}" in WILD_TEMPLATES_PREFIXED[0]:
    ALL_WILD_TEMPLATES = WILD_TEMPLATES_PREFIXED
else:
    for template in WILD_TEMPLATES_PREFIXED:
        for i in range(3):
            copy = template
            copy = copy.replace("{N" + str(i + 1) + "}", "{S1}")
            for k, j in enumerate({1, 2, 3} - {i + 1}):
                copy = copy.replace("{N" + str(j) + "}", "{IO" + str(k + 1) + "}")

            ALL_WILD_TEMPLATES.append(copy)


# %%

# %%

PLACES = [
    "store",
    "garden",
    "restaurant",
    "school",
    "hospital",
    "office",
    "house",
    "station",
]
OBJECTS = [
    "ring",
    "kiss",
    "bone",
    "basketball",
    "computer",
    "necklace",
    "drink",
    "snack",
]

NAMES = [
    "Michael",
    "Christopher",
    "Jessica",
    "Matthew",
    "Ashley",
    "Jennifer",
    "Joshua",
    "Amanda",
    "Daniel",
    "David",
    "James",
    "Robert",
    "John",
    "Joseph",
    "Andrew",
    "Ryan",
    "Brandon",
    "Jason",
    "Justin",
    "Sarah",
    "William",
    "Jonathan",
    "Stephanie",
    "Brian",
    "Nicole",
    "Nicholas",
    "Anthony",
    "Heather",
    "Eric",
    "Elizabeth",
    "Adam",
    "Megan",
    "Melissa",
    "Kevin",
    "Steven",
    "Thomas",
    "Timothy",
    "Christina",
    "Kyle",
    "Rachel",
    "Laura",
    "Lauren",
    "Amber",
    "Richard",
    "Jeffrey",
    "Amy",
    "Crystal",
    "Michelle",
    "Jeremy",
    "Benjamin",
    "Mark",
    "Emily",
    "Aaron",
    "Charles",
    "Rebecca",
    "Jacob",
    "Stephen",
    "Patrick",
    "Sean",
    "Erin",
    "Jamie",
    "Kelly",
    "Nathan",
    "Sara",
    "Paul",
    "Angela",
    "Tyler",
    "Scott",
    "Katherine",
    "Andrea",
    "Gregory",
    "Mary",
    "Travis",
    "Lisa",
    "Kenneth",
    "Bryan",
    "Lindsey",
    "Jose",
    "Alexander",
    "Jesse",
    "Katie",
    "Lindsay",
    "Shannon",
    "Christine",
    "Cody",
    "Allison",
    "Bradley",
    "Samuel",
]

# import json
# import requests
# from ioi_dataset import NAMES

# gender_dict = {}

# for name in NAMES:
#     count_male = 0
#     count_female = 0
#     response = requests.get(f"https://api.genderize.io/?name={name}")
#     data = json.loads(response.text)
#     if data["gender"] == "male":
#         count_male += 1
#     elif data["gender"] == "female":
#         count_female += 1
#     gender_dict[name] = int(count_female > count_male)

# print(gender_dict)
NAMES_GENDER = {
    "Michael": "male",
    "Christopher": "male",
    "Jessica": "female",
    "Matthew": "male",
    "Ashley": "female",
    "Jennifer": "female",
    "Joshua": "male",
    "Amanda": "female",
    "Daniel": "male",
    "David": "male",
    "James": "male",
    "Robert": "male",
    "John": "male",
    "Joseph": "male",
    "Andrew": "male",
    "Ryan": "male",
    "Brandon": "male",
    "Jason": "male",
    "Justin": "male",
    "Sarah": "female",
    "William": "male",
    "Jonathan": "male",
    "Stephanie": "female",
    "Brian": "male",
    "Nicole": "female",
    "Nicholas": "male",
    "Anthony": "male",
    "Heather": "female",
    "Eric": "male",
    "Elizabeth": "female",
    "Adam": "male",
    "Megan": "female",
    "Melissa": "female",
    "Kevin": "male",
    "Steven": "male",
    "Thomas": "male",
    "Timothy": "male",
    "Christina": "female",
    "Kyle": "male",
    "Rachel": "female",
    "Laura": "female",
    "Lauren": "female",
    "Amber": "female",
    "Danielle": "female",
    "Richard": "male",
    "Kimberly": "female",
    "Jeffrey": "male",
    "Amy": "female",
    "Crystal": "female",
    "Michelle": "female",
    "Tiffany": "female",
    "Jeremy": "male",
    "Benjamin": "male",
    "Mark": "male",
    "Emily": "female",
    "Aaron": "male",
    "Charles": "male",
    "Rebecca": "female",
    "Jacob": "male",
    "Stephen": "male",
    "Patrick": "male",
    "Sean": "male",
    "Erin": "female",
    "Jamie": "male",
    "Kelly": "female",
    "Samantha": "female",
    "Nathan": "male",
    "Sara": "female",
    "Dustin": "male",
    "Paul": "male",
    "Angela": "female",
    "Tyler": "male",
    "Scott": "male",
    "Katherine": "female",
    "Andrea": "female",
    "Gregory": "male",
    "Erica": "female",
    "Mary": "female",
    "Travis": "male",
    "Lisa": "female",
    "Kenneth": "male",
    "Bryan": "male",
    "Lindsey": "female",
    "Kristen": "female",
    "Jose": "male",
    "Alexander": "male",
    "Jesse": "male",
    "Katie": "female",
    "Lindsay": "female",
    "Shannon": "female",
    "Vanessa": "female",
    "Courtney": "female",
    "Christine": "female",
    "Alicia": "female",
    "Cody": "male",
    "Allison": "female",
    "Bradley": "male",
    "Samuel": "male",
}


def check_tokenizer(tokenizer):
    safe = True
    for l in [NAMES, PLACES, OBJECTS]:
        for x in l:
            tokens = torch.tensor(tokenizer([" " + x])["input_ids"])[0]
            if len(tokens) > 1:
                print(x, tokens)
                safe = False
    return safe


def get_all_occurence_indices(l, x, prepend_space=True):
    """Get all occurence indices of x in l, with an optional space prepended to x."""
    if prepend_space:
        space_x = " " + x
    else:
        space_x = x
    return [i for i, e in enumerate(l) if e == space_x or e == x]


def names_are_not_distinct(prompt):
    """
    Check that the names in the prompts are distinct.
    """
    if "IO2" in prompt:
        return (
            prompt["IO1"] == prompt["IO2"]
            or prompt["IO1"] == prompt["S"]
            or prompt["IO2"] == prompt["S"]
        )
    else:
        return prompt["IO"] == prompt["S"]


PromptType = Literal["mixed", "ABBA", "BABA"]


def get_wild_template_order(template):
    s1 = template.index("{S1}")
    io1 = template.index("{IO1}")
    io2 = template.index("{IO2}")
    if s1 < io1 and s1 < io2:
        return "S1IO1IO2"
    elif io1 < s1 and s1 < io2:
        return "IO1S1IO2"
    else:
        return "IO1IO2S1"


class IOIDataset:
    """Inspired by https://github.com/redwoodresearch/Easy-Transformer/blob/main/easy_transformer/ioi_dataset.py,
    but not the same."""

    prompt_type: PromptType
    word_idx: dict[
        str, torch.Tensor
    ]  # keys depend on the prompt family, value is tensor

    def __init__(
        self,
        N,
        prompt_type: PromptType = "mixed",
        prompt_family: Literal["IOI", "ABC"] = "IOI",
        nb_templates=None,  # if not None, limit the number of templates to use
        add_prefix_space=True,
        device="cuda:0",
        manual_metadata=None,
        seed=42,
        wild_template=False,
        nb_names=None,
        tokenizer=None,
    ):
        self.seed = seed
        random.seed(seed)
        self.nb_names = nb_names
        self.names = NAMES[: self.nb_names]
        self.N = N
        self.device = device
        self.add_prefix_space = add_prefix_space
        self.prompt_type = prompt_type
        self.prompt_family = prompt_family  # can change to "ABC" after flipping names

        if manual_metadata is not None:  # we infer the family from the metadata
            if "IO2" in manual_metadata[0].keys():
                self.prompt_family = "ABC"
            else:
                self.prompt_family = "IOI"

        self.wild_template = wild_template
        if wild_template:
            self.templates = ALL_WILD_TEMPLATES
            self.prompt_family = "wild"
            nb_templates = len(self.templates)
        else:
            if nb_templates is None:
                if prompt_type == "mixed":
                    nb_templates = len(BABA_TEMPLATES) * 2
                else:
                    nb_templates = len(BABA_TEMPLATES)
            if prompt_type == "ABBA":
                self.templates = ABBA_TEMPLATES[:nb_templates].copy()
            elif prompt_type == "BABA":
                self.templates = BABA_TEMPLATES[:nb_templates].copy()
            elif prompt_type == "mixed":
                self.templates = (
                    BABA_TEMPLATES[: (nb_templates // 2) + (nb_templates % 2)].copy()
                    + ABBA_TEMPLATES[: nb_templates // 2].copy()
                )

        assert not (
            (prompt_type == "mixed" and nb_templates % 2 != 0) and not wild_template
        ), "Mixed dataset with odd number of templates!"

        self.nb_templates = nb_templates

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = 50256
            self.tokenizer.add_prefix_space = add_prefix_space  # type: ignore

        self.initialize_prompts(manual_metadata=manual_metadata)
        self.initialize_word_idx()

        if self.prompt_family == "IOI":
            self.io_tokenIDs = self.prompts_tok[torch.arange(N), self.word_idx["IO"]]
            self.s_tokenIDs = self.prompts_tok[torch.arange(N), self.word_idx["S1"]]
        elif self.prompt_family == "ABC":
            self.io1_tokenIDs = self.prompts_tok[torch.arange(N), self.word_idx["IO1"]]
            self.io2_tokenIDs = self.prompts_tok[torch.arange(N), self.word_idx["IO2"]]
            self.s_tokenIDs = self.prompts_tok[torch.arange(N), self.word_idx["S"]]
        elif self.prompt_family == "wild":
            self.io_tokenIDs = self.prompts_tok[torch.arange(N), self.word_idx["IO1"]]
            self.s_tokenIDs = self.prompts_tok[torch.arange(N), self.word_idx["S1"]]
            self.io2_tokenIDs = self.prompts_tok[torch.arange(N), self.word_idx["IO2"]]

    def initialize_prompts(self, manual_metadata=None):
        # define the prompts' metadata

        if manual_metadata is None:
            self.prompts_metadata = []
            for i in range(self.N):
                template_idx = random.choice(list(range(len(self.templates))))
                s = random.choice(self.names)
                io = random.choice(self.names)
                while io == s:
                    io = random.choice(self.names)
                place = random.choice(PLACES)
                obj = random.choice(OBJECTS)
                self.prompts_metadata.append(
                    {
                        "S": s,
                        "IO": io,
                        "TEMPLATE_IDX": template_idx,
                        "[PLACE]": place,
                        "[OBJECT]": "none"
                        if "said" in self.templates[template_idx]
                        else obj,
                        "order": "ABB"
                        if self.templates[template_idx] in ABBA_TEMPLATES
                        else "BAB",
                    }
                )
                if self.wild_template:
                    self.prompts_metadata[-1]["order"] = get_wild_template_order(
                        self.templates[template_idx]
                    )
                    self.prompts_metadata[-1]["IO1"] = self.prompts_metadata[-1]["IO"]
                    io2 = random.choice(self.names)
                    while io2 == s or io2 == self.prompts_metadata[-1]["IO"]:
                        io2 = random.choice(self.names)
                    self.prompts_metadata[-1]["IO2"] = io2

        else:
            self.prompts_metadata = manual_metadata

        # define the prompts' texts
        self.prompts_text = []
        for metadata in self.prompts_metadata:
            cur_template = self.templates[metadata["TEMPLATE_IDX"]]
            if self.prompt_family == "IOI":
                self.prompts_text.append(
                    cur_template.format(
                        IO=metadata["IO"],
                        S1=metadata["S"],
                        S2=metadata["S"],
                        PLACE=metadata["[PLACE]"],
                        OBJECT=metadata["[OBJECT]"],
                    )
                )
            elif self.prompt_family == "ABC":
                self.prompts_text.append(
                    cur_template.format(
                        IO=metadata["IO1"],
                        S1=metadata["IO2"],
                        S2=metadata["S"],
                        PLACE=metadata["[PLACE]"],
                        OBJECT=metadata["[OBJECT]"],
                    )
                )
            elif self.prompt_family == "wild":
                self.prompts_text.append(
                    cur_template.format(
                        IO1=metadata["IO1"],
                        IO2=metadata["IO2"],
                        S1=metadata["S"],
                        S2=metadata["S"],
                        PLACE=metadata["[PLACE]"],
                        OBJECT=metadata["[OBJECT]"],
                    )
                )
            else:
                raise ValueError("Unknown prompt family")

        # define the tokens
        self.prompts_tok = torch.tensor(
            self.tokenizer(self.prompts_text, padding=True)["input_ids"]
        )
        self.prompts_tok.to(self.device)

        # to get the position of the relevant names in the _tokenized_ sentences, we split the text sentences
        # by tokens, and we replace the S1, S2 IO (IO1, IO2 and S in ABC) by their annotations.

        self.prompts_text_toks = [
            [
                self.tokenizer.decode([x])
                for x in self.tokenizer(self.prompts_text[j])["input_ids"]  # type: ignore
            ]
            for j in range(len(self))
        ]

        for i in range(len(self)):
            s_idx = get_all_occurence_indices(
                self.prompts_text_toks[i], self.prompts_metadata[i]["S"]
            )
            if self.prompt_family == "IOI":
                io_idx = get_all_occurence_indices(
                    self.prompts_text_toks[i], self.prompts_metadata[i]["IO"]
                )[0]
                assert len(s_idx) == 2
                self.prompts_text_toks[i][s_idx[0]] = "{S1}"
                self.prompts_text_toks[i][s_idx[1]] = "{S2}"
                self.prompts_text_toks[i][io_idx] = "{IO}"
            elif self.prompt_family == "ABC":
                io1_idx = get_all_occurence_indices(
                    self.prompts_text_toks[i], self.prompts_metadata[i]["IO1"]
                )[0]
                io2_idx = get_all_occurence_indices(
                    self.prompts_text_toks[i], self.prompts_metadata[i]["IO2"]
                )[0]
                self.prompts_text_toks[i][io1_idx] = "{IO1}"
                self.prompts_text_toks[i][io2_idx] = "{IO2}"
                self.prompts_text_toks[i][s_idx[0]] = "{S}"
            elif self.prompt_family == "wild":
                assert (
                    len(s_idx) == 2
                ), f"len(s_idx) = {len(s_idx)} for {self.prompts_text[i]} {self.prompts_metadata[i]}"
                io1_idx = get_all_occurence_indices(
                    self.prompts_text_toks[i], self.prompts_metadata[i]["IO1"]
                )[0]
                io2_idx = get_all_occurence_indices(
                    self.prompts_text_toks[i], self.prompts_metadata[i]["IO2"]
                )[0]
                self.prompts_text_toks[i][io1_idx] = "{IO1}"
                self.prompts_text_toks[i][io2_idx] = "{IO2}"
                self.prompts_text_toks[i][s_idx[0]] = "{S1}"
                self.prompts_text_toks[i][s_idx[1]] = "{S2}"

    def initialize_word_idx(self):
        self.word_idx = {}

        if self.prompt_family == "IOI":
            literals = ["{IO}", "{S1}", "{S2}"]
        elif self.prompt_family == "ABC":
            literals = ["{IO1}", "{IO2}", "{S}"]  # disjoint set of literals
        elif self.prompt_family == "wild":
            literals = ["{IO1}", "{IO2}", "{S1}", "{S2}"]
        else:
            raise ValueError("Unknown prompt family")

        for word in literals:
            self.word_idx[word[1:-1]] = torch.tensor(
                [self.prompts_text_toks[i].index(word) for i in range(len(self))]
            )

        if self.prompt_family == "IOI" or self.prompt_family == "wild":
            self.word_idx["S1+1"] = self.word_idx["S1"] + 1
        elif self.prompt_family == "ABC":
            self.word_idx["IO1+1"] = (
                self.word_idx["IO1"] + 1
            )  # here to be able to compare

        self.word_idx["END"] = torch.tensor(
            [len(self.prompts_text_toks[i]) - 1 for i in range(len(self))]
        )

    def gen_flipped_prompts(self, flip: str) -> "IOIDataset":
        """
        Return a IOIDataset where the name to flip has been replaced by a random name.
        """
        assert flip in ["IO", "S1", "S2", "IO2", "IO1", "S", "order"], "Unknown flip"
        assert (
            flip in ["IO", "S1", "S2", "order", "S"] and self.prompt_family == "IOI"
        ) or (
            flip in ["IO1", "IO2", "S", "order"] and self.prompt_family == "ABC"
        ), f"{flip} is illegal for prompt family {self.prompt_family}"

        new_prompts_metadata = deepcopy(self.prompts_metadata)

        new_prompt_type = "no type"
        if flip in ["IO", "IO1", "IO2", "S"]:  # when the flip keeps
            for prompt in new_prompts_metadata:
                prompt[flip] = random.choice(self.names)
                while names_are_not_distinct(prompt):
                    prompt[flip] = random.choice(self.names)
            new_family = self.prompt_family
            new_prompt_type = self.prompt_type
        elif flip == "S1":
            for prompt in new_prompts_metadata:
                prompt["IO2"] = prompt[
                    "IO"
                ]  # this lead to a change in prompt family from IOI to ABC. S stays the same.
                prompt["IO1"] = prompt["IO"]
                del prompt["IO"]
                prompt["IO2"] = random.choice(self.names)
                while names_are_not_distinct(prompt):
                    prompt["IO2"] = random.choice(self.names)
            new_family = "ABC"
            new_prompt_type = self.prompt_type
        elif flip == "S2":
            for prompt in new_prompts_metadata:
                prompt["IO2"] = prompt["S"]
                prompt["IO1"] = prompt["IO"]
                del prompt["IO"]
                prompt["S"] = random.choice(self.names)
                while names_are_not_distinct(prompt):
                    prompt["S"] = random.choice(self.names)
            new_family = "ABC"
            new_prompt_type = self.prompt_type

        elif flip == "order":
            if self.prompt_family == "IOI":
                for prompt in new_prompts_metadata:
                    prompt["TEMPLATE_IDX"] = find_flipped_template_idx(
                        prompt["TEMPLATE_IDX"], self.prompt_type, self.nb_templates
                    )

                if self.prompt_type == "ABBA":
                    new_prompt_type = "BABA"
                elif self.prompt_type == "BABA":
                    new_prompt_type = "ABBA"
                elif self.prompt_type == "mixed":
                    new_prompt_type = self.prompt_type

            if self.prompt_family == "ABC":
                new_prompt_type = self.prompt_type
                raise NotImplementedError()
                # TODO: change the order of the first two names in the prompt!

            new_family = self.prompt_type
        else:
            raise NotImplementedError()

        assert new_prompt_type != "no type", "new prompt type not set"

        return IOIDataset(
            N=self.N,
            prompt_type=new_prompt_type,
            manual_metadata=new_prompts_metadata,
            prompt_family=new_family,  # type: ignore # TBD: fix
            nb_templates=self.nb_templates,
            add_prefix_space=self.add_prefix_space,
            device=self.device,
        )

    def __len__(self):
        return self.N


def find_flipped_template_idx(temp_idx, prompt_type, nb_templates):
    """Given a template index and the prompt type of a dataset, return the indice of the flipped template in the new dataset. This relies on the fact that the templates for the object are preserving the order from ABBA_TEMPLATES and BABA_TEMPLATES"""
    if prompt_type in ["ABBA", "BABA"]:
        return temp_idx
    elif prompt_type == "mixed":
        if temp_idx < nb_templates // 2:
            return nb_templates // 2 + temp_idx
        else:
            return temp_idx - nb_templates // 2


# %%


# %%
