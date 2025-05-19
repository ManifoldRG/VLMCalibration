"""
Answer validators for all datasets used in calibration experiments.
These validators are shared between local and vLLM calibration scripts.
"""

from typing import Optional, Any
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client if API key is available
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

def validate_gsm8k_answer(answer: Optional[int], true_answer: int) -> Optional[bool]:
    """Validate GSM8K answers."""
    if answer is None:
        return None
    return abs(answer - true_answer) < 1e-4

def validate_mmlu_answer(answer: Optional[str], true_answer: str) -> Optional[bool]:
    """Validate MMLU answers."""
    return None if answer is None else answer == true_answer

def validate_medmcqa_answer(answer: Optional[str], true_answer: str) -> Optional[bool]:
    """Validate MedMCQA answers."""
    return None if answer is None else answer == true_answer

def validate_simpleqa_answer(question: str, model_answer: str, true_answer: str) -> Optional[bool]:
    """Grade SimpleQA answers with GPT-4-o."""
    if openai_client is None:
        return None

    try:
        from simpleQA_grader_template import GRADER_TEMPLATE

        grading_prompt = GRADER_TEMPLATE.format(
            question=question, target=true_answer, predicted_answer=model_answer
        )

        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful grading assistant."},
                {"role": "user", "content": grading_prompt},
            ],
            temperature=0.7,
            max_tokens=1,
        )
        letter = resp.choices[0].message.content.strip()
        if letter == "A":
            return True
        if letter == "B":
            return False
        if letter == "C":
            return "NOT_ATTEMPTED"  # type: ignore[return-value]
    except Exception as exc:
        print(f"Grading error: {exc}")

    return None 