"""
Prompt templates for all datasets used in calibration experiments.
These templates are shared between local and vLLM calibration scripts.
"""

def format_gsm8k_question(question: str) -> str:
    return (
        "Given the following problem, reason and give a final answer to the problem.\n"
        f"Problem: {question}\n"
        'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.'
    )

def format_mmlu_question(question: str, choices_list: list, subject: str) -> str:
    formatted_choices = "\n".join([f"{chr(65 + i)}: {choice}" for i, choice in enumerate(choices_list)])
    return (
        f"The following is a multiple choice question (with choices) about {subject}.\n"
        f"Reason and give a final answer to the question. Your response should end with 'The final answer is [answer]' where [answer] is solely the letter corresponding to the correct choice.\n"
        f"Question: {question}\n"
        f"Choices:\n{formatted_choices}"
    )

def format_medmcqa_question(question: str, choices_list: list) -> str:
    formatted_choices = "\n".join([f"{chr(65 + i)}: {choice}" for i, choice in enumerate(choices_list)])
    return (
        f"Given the following medical question and options, reason and select the correct answer letter (A-D).\n"
        f"Question: {question}\n"
        f"Options:\n{formatted_choices}\n"
        f"Your response should end with 'The final answer is [letter]' where [letter] is A, B, C or D."
    )

def format_simpleqa_question(question: str) -> str:
    return (
        "Given the following question, provide a clear and concise answer.\n"
        f"Question: {question}\n\n"
        'Your response should end with "The answer is: [answer]" where [answer] is your complete answer.'
    ) 