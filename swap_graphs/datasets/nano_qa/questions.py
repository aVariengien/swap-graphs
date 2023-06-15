# %%
from typing import List, Dict, Any

QUESTIONS = [
    # character name
    {
        "question": "What's the name of the main character?",
        "answer_prefix": "The main character is named",
        "querried_variable": "character_name",
    },
    {
        "question": "Who is the protagonist of the story?",
        "answer_prefix": "The protagonist's name is",
        "querried_variable": "character_name",
    },
    {
        "question": "Which character serves as the story's main focus?",
        "answer_prefix": "The primary focus of the story is on a character named",
        "querried_variable": "character_name",
    },
    # city
    {
        "question": "Where does the story take place?",
        "answer_prefix": "The story takes place in the city called",
        "querried_variable": "city",
    },
    {
        "question": "In which city is the plot set?",
        "answer_prefix": "The plot of the story takes place in the city of",
        "querried_variable": "city",
    },
    {
        "question": "Where is the story located?",
        "answer_prefix": "The story is located in a city named",
        "querried_variable": "city",
    },
    # occupation
    {
        "question": "What job does the main character have?",
        "answer_prefix": "The main character is a professional",
        "querried_variable": "character_occupation",
    },
    {
        "question": "In which profession is the central character involved?",
        "answer_prefix": "The central character is a professional",
        "querried_variable": "character_occupation",
    },
    {
        "question": "Which vocation does the protagonist pursue?",
        "answer_prefix": "The protagonist is a professional",
        "querried_variable": "character_occupation",
    },
    # time of the day
    {
        "question": "At what time of day is the story set?",
        "answer_prefix": "The story is set in the",
        "querried_variable": "day_time",
    },
    {
        "question": "During which part of the day does the story take place?",
        "answer_prefix": "The story takes place in the",
        "querried_variable": "day_time",
    },
    {
        "question": "At what time does the story occur?",
        "answer_prefix": "The story occurs in the",
        "querried_variable": "day_time",
    },
    # season
    {
        "question": "Which season is it in the story?",
        "answer_prefix": "The story takes place in the",
        "querried_variable": "season",
    },
    {
        "question": "What is the season during which the story takes place?",
        "answer_prefix": "The story takes place in the",
        "querried_variable": "season",
    },
    {
        "question": "In what season does the story occur?",
        "answer_prefix": "The story occurs in the",
        "querried_variable": "season",
    },
]


QUESTION_PROMPT = """<|endoftext|>

Here is a short story. Read it carefully and answer the questions below.

{nanostory_text}

Answer the questions below, The answers should be concise and to the point.

Question: {question}

Answer: {answer_prefix}"""


def gen_question_prompt(nanostory_text: str, question_dict: Dict[str, str]):
    return QUESTION_PROMPT.format(
        nanostory_text=nanostory_text,
        question=question_dict["question"],
        answer_prefix=question_dict["answer_prefix"],
    )


# %%
