import numpy as np
from datasets import load_dataset
import re
import pandas as pd
from tqdm.auto import tqdm
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
import traceback
from simpleQA_grader_template import GRADER_TEMPLATE
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# OpenAI configuration for SimpleQA grading
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Server configuration
url = "http://localhost:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}

# Dataset-specific question formatters
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

# Dataset-specific answer parsers
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

def validate_simpleqa_answer(question: str, model_answer: str, true_answer: str) -> bool:
    """Use GPT-4 to grade the answer by comparing model's answer with true answer"""
    grading_prompt = GRADER_TEMPLATE.format(question=question, target=true_answer, predicted_answer=model_answer)

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful grading assistant."},
                {"role": "user", "content": grading_prompt}
            ],
            temperature=0.7,
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
        print(f"Grading error: {e}")
        return None

def send_request(payload):
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Request error: {e}")
        return None

def process_single_question(idx: int, dataset_type: str, dataset, dataset_split: str):
    try:
        
        if dataset_type == "simpleqa":
            question = dataset[dataset_split][idx]["problem"]
        else:
            question = dataset[dataset_split][idx]["question"]
        
        if dataset_type == "gsm8k":
            question_str = format_gsm8k_question(question)
        elif dataset_type == "mmlu":
            choices_list = dataset[dataset_split][idx]["choices"]
            subject = dataset[dataset_split][idx]["subject"].replace("_", " ")
            question_str = format_mmlu_question(question, choices_list, subject)
        elif dataset_type == "medmcqa":
            choices_list = [
                dataset[dataset_split][idx]["opa"],
                dataset[dataset_split][idx]["opb"],
                dataset[dataset_split][idx]["opc"],
                dataset[dataset_split][idx]["opd"]
            ]
            question_str = format_medmcqa_question(question, choices_list)
        elif dataset_type == "simpleqa":
            question_str = format_simpleqa_question(question)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        # First request for getting the answer
        first_payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question_str},
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
        }

        first_response = send_request(first_payload)
        if not first_response:
            return None

        response_text = first_response["choices"][0]["message"]["content"]
        
        # Parse answer based on dataset type
        if dataset_type == "gsm8k":
            answer = parse_gsm8k_answer(response_text)
        elif dataset_type == "mmlu":
            answer = parse_mmlu_answer(response_text)
        elif dataset_type == "medmcqa":
            answer = parse_medmcqa_answer(response_text)
        elif dataset_type == "simpleqa":
            answer = parse_simpleqa_answer(response_text)

        # Add chain-of-thought injection for cot_exp
        if args.exp_type == "cot_exp":
            inject_cot_payload = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question_str},
                    {"role": "assistant", "content": response_text},
                    {
                        "role": "user",
                        "content": "Before answering whether your above answer is correct, please provide a detailed chain-of-thought explanation of your reasoning. Explain step-by-step how you arrived at your answer and why you think it is correct or might be incorrect.",
                    },
                ],
                "temperature": 0.7,
                "max_tokens": 1024,
            }

            inject_cot_response = send_request(inject_cot_payload)
            if not inject_cot_response:
                return None
            inject_cot_text = inject_cot_response["choices"][0]["message"]["content"]
        else:
            inject_cot_text = None

        # Final request for confidence check with logprobs
        final_payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question_str},
                {"role": "assistant", "content": response_text},
            ],
            "temperature": 0.7,
            "max_tokens": 1,
            "n_probs": 25,
        }

        # Add CoT messages if in cot_exp mode
        if args.exp_type == "cot_exp":
            final_payload["messages"].extend([
                {
                    "role": "user",
                    "content": "Before answering whether your above answer is correct, please provide a detailed chain-of-thought explanation of your reasoning. Explain step-by-step how you arrived at your answer and why you think it is correct or might be incorrect.",
                },
                {"role": "assistant", "content": inject_cot_text},
            ])

        final_payload["messages"].append({
            "role": "user",
            "content": "Is the above answer correct? Answer only with the single word 'true' or 'false'.",
        })

        final_response = send_request(final_payload)
        if not final_response:
            return None

        # Extract logprobs for 'true' and 'false' tokens
        logprobs = final_response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        probs = {"true": 0.0, "false": 0.0}
        
        for item in logprobs:
            token = item["token"].lower()
            if any(key in token for key in ["true", "false", "correct", "incorrect", "wrong",
                                          "truth", "yes", "right", "verdade",  # True variations
                                          "fake", "no", "not", "none"]):      # False variations
                if ("true" in token or "correct" in token or "truth" in token or 
                    "yes" in token or "right" in token or "verdade" in token):
                    actual_key = "true"
                elif ("false" in token or "incorrect" in token or "wrong" in token or 
                      "fake" in token or "no" in token or "not" in token or "none" in token):
                    actual_key = "false"
                probs[actual_key] += np.exp(item["logprob"])

        p_true = probs["true"] / (probs["true"] + probs["false"])

        # Get and validate true answer based on dataset type
        if dataset_type == "gsm8k":
            true_answer = int(dataset[dataset_split][idx]["answer"].split("\n")[-1][5:].replace(",", ""))
            is_correct = validate_gsm8k_answer(answer, true_answer)
        elif dataset_type == "mmlu":
            true_answer_idx = dataset[dataset_split][idx]["answer"]
            true_answer = ["A", "B", "C", "D"][true_answer_idx]
            is_correct = validate_mmlu_answer(answer, true_answer)
        elif dataset_type == "medmcqa":
            true_answer = chr(65 + dataset[dataset_split][idx]["cop"])
            is_correct = validate_medmcqa_answer(answer, true_answer)
        elif dataset_type == "simpleqa":
            true_answer = dataset[dataset_split][idx]["answer"]
            is_correct = validate_simpleqa_answer(question, answer, true_answer)

        result = {
            "question": question,
            "response": response_text,
            "answer": answer,
            "p_true": p_true,
            "true_answer": true_answer,
            "correct": is_correct,
        }
        if args.exp_type == "cot_exp":
            result["inject_cot"] = inject_cot_text
        return result
    except Exception as e:
        print(f"Error processing question {idx}: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        choices=["llama", "qwen", "gemma"],
        help="Name of local model to use",
    )
    parser.add_argument(
        "dataset_name",
        choices=["gsm8k", "mmlu", "medmcqa", "simpleqa"],
        help="Dataset to use",
    )
    parser.add_argument(
        "dataset_split",
        choices=["train", "test", "validation"],
        help="Dataset split to use",
    )
    parser.add_argument(
        "exp_type",
        choices=["cot_exp", "zs_exp"],
        help="Experiment type",
    )
    parser.add_argument(
        "max_workers",
        type=int,
        default=12,
        help="Maximum number of workers to use",
    )
    args = parser.parse_args()

    # Load appropriate dataset
    assert not (args.dataset_name == "medmcqa" and args.dataset_split == "test"), "MedMCQA test split does not contain answers - cannot calibrate now"

    
    if args.dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main")
    elif args.dataset_name == "mmlu":
        dataset = load_dataset("cais/mmlu", "all")
    elif args.dataset_name == "medmcqa":
        dataset = load_dataset("openlifescienceai/medmcqa", "default")
    elif args.dataset_name == "simpleqa":
        dataset = load_dataset("basicv8vc/SimpleQA")



    dataset_split = args.dataset_split
    print(f"Dataset: {args.dataset_name}, Split: {dataset_split}")
    
    BASE_DIR = f"{args.exp_type}/{args.dataset_name}"
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"Base directory: {BASE_DIR}")

    records = []
    start_idx = 0
    end_idx = len(dataset[dataset_split])

    if args.dataset_name == "medmcqa" and args.dataset_split == "train":
        np.random.seed(42)
        end_idx = min(5000, len(dataset[dataset_split]))
        indices = [int(i) for i in np.random.choice(len(dataset[dataset_split]), end_idx, replace=False)]
        start_idx = 0

    FINAL_FILE_NAME = f"{BASE_DIR}/{args.exp_type}_records_{args.dataset_split}_full_{args.model_name}.csv"
    assert not os.path.exists(FINAL_FILE_NAME), "Final file already exists - You should delete it before running the experiment"

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        if args.dataset_name == "medmcqa" and args.dataset_split == "train":
            for idx in indices:
                futures.append(executor.submit(process_single_question, idx, args.dataset_name, dataset, dataset_split))
        else:
            for idx in range(start_idx, end_idx):
                futures.append(executor.submit(process_single_question, idx, args.dataset_name, dataset, dataset_split))

        for idx, future in enumerate(tqdm(as_completed(futures), total=end_idx - start_idx, desc="Processing dataset")):
            result = future.result()
            if result:
                records.append(result)

            if (len(records) > 0) and (len(records) % 50 == 0):
                pd.DataFrame.from_records(records).to_csv(
                    f"{BASE_DIR}/{args.exp_type}_records_{args.dataset_split}_partial_{args.model_name}.csv"
                )

    # Final save
    final_df = pd.DataFrame.from_records(records)
    final_df.to_csv(FINAL_FILE_NAME)

    # Delete the partial results file after final save
    partial_file = f"{BASE_DIR}/{args.exp_type}_records_{args.dataset_split}_partial_{args.model_name}.csv"
    if os.path.exists(partial_file):
        os.remove(partial_file) 