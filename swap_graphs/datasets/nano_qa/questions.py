# %%
QUESTIONS = [
    {
        "question": "What's the name of the main character?",
        "answer_prefix": "The main character is named",
        "querried_variable": "character_name",
    },
    {
        "question": "Where does the story take place?",
        "answer_prefix": "The story takes place in the city called",
        "querried_variable": "city",
    },
    {
        "question": "What job does the main character have?",
        "answer_prefix": "The main character is a professional",
        "querried_variable": "character_occupation",
    },
    {
        "question": "At what time of day is the story set?",
        "answer_prefix": "The story is set in the",
        "querried_variable": "day_time",
    },
    {
        "question": "Which season is it in the story?",
        "answer_prefix": "The story takes place in the",
        "querried_variable": "season",
    },
]


QUESTION_PROMPT = """<|endoftext|>

Here is a short story. Read it carefully and answer the questions below.

{nanostory_text}

Answer the questions below, The answers should be concise and to the point.

Question: {question}

Answer: {answer_prefix}"""


def gen_question_prompt(nanostory_text: str, question_dict: dict[str, str]):
    return QUESTION_PROMPT.format(
        nanostory_text=nanostory_text,
        question=question_dict["question"],
        answer_prefix=question_dict["answer_prefix"],
    )


# %%
