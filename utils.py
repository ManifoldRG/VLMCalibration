"""
Dataset-specific question formatters, parsers and validators
"""
import re
from simpleQA_grader_template import GRADER_TEMPLATE
from openai import OpenAI
import time

def format_gsm8k_question(question: str) -> str:
    return f'Given the following problem, reason and give a final answer to the problem.\nProblem: {question}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.'

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
    return f"""Given the following question, provide a clear and concise answer.
Question: {question}

Your response should end with "The answer is: [answer]" where [answer] is your complete answer."""


def parse_gsm8k_answer(output: str) -> int:
    final_answer_match = re.search(r"The final answer is \$?(\d+,?\d*)", output)
    if final_answer_match:
        return int(final_answer_match.group(1).replace(",", ""))
    matches = re.findall(r"\$?(\d+,?\d*)", output)
    return int(matches[-1].replace(",", "")) if matches else None

def parse_mmlu_answer(output: str) -> str:
    final_answer_match = re.search(r"The final answer is ([A-D])", output)
    if final_answer_match:
        return final_answer_match.group(1)
    return None

def parse_medmcqa_answer(output: str) -> str:
    final_answer_match = re.search(r"The final answer is ([A-D])", output)
    if final_answer_match:
        return final_answer_match.group(1)
    return None

def parse_simpleqa_answer(output: str) -> str:
    answer_match = re.search(r"The answer is: (.*?)(?:\.|$)", output, re.IGNORECASE | re.MULTILINE)
    if answer_match:
        return answer_match.group(1).strip()
    return output

def parse_verbalized_confidence(output: str) -> float:
    """Parse verbalized confidence score from model output.
    
    Expected format: "Confidence: <number between 0.0 and 1.0>"
    Returns the confidence score as a float, or None if not found.
    """
    confidence_match = re.search(r"Confidence:\s*([0-9]*\.?[0-9]+)", output, re.IGNORECASE)
    if confidence_match:
        try:
            confidence = float(confidence_match.group(1))
            # Clamp to valid range [0.0, 1.0]
            return max(0.0, min(1.0, confidence))
        except ValueError:
            return None
    return None

# Dataset-specific answer validators
def validate_gsm8k_answer(answer: int, true_answer: int) -> bool:
    if answer is None or not isinstance(true_answer, (int, float)):
        return None
    return abs(answer - true_answer) < 0.0001

def validate_mmlu_answer(answer: str, true_answer: str) -> bool:
    if answer is None:
        return None
    return answer == true_answer

def validate_medmcqa_answer(answer: str, true_answer: str) -> bool:
    if answer is None:
        return None
    return answer == true_answer
def validate_simpleqa_answer(question: str, model_answer: str, true_answer: str, client=None) -> bool:
    """Use GPT-4 to grade the answer by comparing model's answer with true answer"""

    if client is None:
        raise ValueError("Client is not set for simpleQA grading")

    grading_prompt = GRADER_TEMPLATE.format(question=question, target=true_answer, predicted_answer=model_answer)

    retries = 5
    delay = 1  # Initial delay in seconds
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful grading assistant."},
                    {"role": "user", "content": grading_prompt}
                ],
                temperature=0.1,
                max_tokens=1,
            )
            grade_letter = response.choices[0].message.content.strip()
            is_correct = grade_letter == "A"
            is_incorrect = grade_letter == "B"
            is_not_attempted = grade_letter == "C"
            
            if is_correct:
                return True
            elif is_incorrect:
                return False
            elif is_not_attempted:
                return "NOT_ATTEMPTED"

        except Exception as e:
            if attempt == retries - 1:  # Last attempt
                print(f"Grading error after {retries} attempts: {e}")
                return None
                
            print(f"Grading attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
