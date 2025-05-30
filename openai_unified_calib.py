import numpy as np
from datasets import load_dataset
import re
import pandas as pd
from tqdm.auto import tqdm
import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from dotenv import load_dotenv
import os
import traceback

from utils import *

load_dotenv()

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)


def get_safe_model_name(model_name: str) -> str:
    # Replace slashes and other special chars with underscores
    model_name = model_name.split("/")[-1]
    safe_name = re.sub(r"[/\\:]", "_", model_name)
    # Remove any other non-alphanumeric chars except underscores and dashes
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "", safe_name)
    return safe_name


TRUE_SYNONYMS = [
    "true",
    "correct",
    "truth",
    "yes",
    "right",
    "verdade",
]

FALSE_SYNONYMS = [
    "false",
    "incorrect",
    "wrong",
    "fake",
    "no",
    "not",
    "none",
]


# ---------------------------------------------------------------------------
# Dataset-specific formatting helpers (delegated to utils.py if available)
# ---------------------------------------------------------------------------

def format_question(task):
    try:
        idx = task["idx"]
        dataset_type = task["dataset_type"]
        dataset = task["dataset"]
        dataset_split = task["dataset_split"]

        if dataset_type == "simpleqa":
            question = dataset[dataset_split][idx]["problem"]
        elif dataset_type == "truthfulqa":
            question = dataset[dataset_split][idx]["question"]
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
                dataset[dataset_split][idx]["opd"],
            ]
            question_str = format_medmcqa_question(question, choices_list)
        elif dataset_type == "simpleqa":
            question_str = format_simpleqa_question(question)
        elif dataset_type == "truthfulqa":
            question_str = format_truthfulqa_question(question)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        task["question"] = question
        task["question_str"] = question_str
        return task
    except Exception as e:
        print(f"Error formatting question {idx}: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Inference helpers – all API calls target OpenAI
# ---------------------------------------------------------------------------

def get_initial_response(task):
    try:
        model_name = task["model_name"]
        temp = task["temp"]
        max_tokens = task["max_tokens"]
        question_str = task["question_str"]

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question_str},
            ],
            temperature=temp,
            max_tokens=max_tokens,
        )

        response_text = response.choices[0].message.content
        task["response_text"] = response_text
        return task
    except Exception as e:
        print(f"Error getting initial response for task: {e}")
        traceback.print_exc()
        return None


def get_cot_explanation(task):
    try:
        if task["exp_type"] not in ["cot_exp", "verbalized_cot"]:
            return task

        model_name = task["model_name"]
        temp = task["temp"]
        max_tokens = task["max_tokens"]
        question_str = task["question_str"]
        response_text = task["response_text"]

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question_str},
                {"role": "assistant", "content": response_text},
                {
                    "role": "user",
                    "content": (
                        "Before answering whether your above answer is correct, "
                        "please provide a detailed chain-of-thought explanation of "
                        "your reasoning. Explain step-by-step how you arrived at "
                        "your answer and why you think it is correct or might be "
                        "incorrect."
                    ),
                },
            ],
            temperature=temp,
            max_tokens=max_tokens,
        )

        cot_text = response.choices[0].message.content
        task["cot_text"] = cot_text
        return task
    except Exception as e:
        print(f"Error getting CoT explanation: {e}")
        traceback.print_exc()
        return None


def get_confidence_score(task):
    """Ask the model to self-evaluate and extract token-level log-probs."""
    try:
        model_name = task["model_name"]
        temp = task["temp"]
        question_str = task["question_str"]
        response_text = task["response_text"]
        exp_type = task["exp_type"]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question_str},
            {"role": "assistant", "content": response_text},
        ]

        if exp_type == "cot_exp":
            cot_text = task["cot_text"]
            messages.extend([
                {
                    "role": "user",
                    "content": (
                        "Before answering whether your above answer is correct, "
                        "please provide a detailed chain-of-thought explanation of "
                        "your reasoning. Explain step-by-step how you arrived at "
                        "your answer and why you think it is correct or might be "
                        "incorrect."
                    ),
                },
                {"role": "assistant", "content": cot_text},
            ])

        messages.append({
            "role": "user",
            "content": "Is the above answer correct? Answer only with the single word 'true' or 'false'.",
        })

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temp,
            max_tokens=1,
            logprobs=True,
            top_logprobs=20
        )

        # Extract logprobs from the response
        if response.choices and len(response.choices) > 0:
            choice = response.choices[0]
            if hasattr(choice, 'logprobs') and choice.logprobs and choice.logprobs.content:
                # Get the logprobs for the tokens
                logprobs = []
                for token_info in choice.logprobs.content:
                    if hasattr(token_info, 'top_logprobs') and token_info.top_logprobs:
                        for top_prob in token_info.top_logprobs:
                            token = top_prob.token.lower()
                            if any(syn in token for syn in TRUE_SYNONYMS):
                                logprobs.append({
                                    "token": "true",
                                    "logprob": top_prob.logprob
                                })
                            elif any(syn in token for syn in FALSE_SYNONYMS):
                                logprobs.append({
                                    "token": "false", 
                                    "logprob": top_prob.logprob
                                })
                
                task["logprobs"] = logprobs
                return task

        print(f"Unexpected response structure")
        return None

    except Exception as e:
        print(f"Error getting confidence score: {e}")
        traceback.print_exc()
        return None


def get_verbalized_confidence(task):
    """Ask the model to provide a verbalized confidence score."""
    try:
        model_name = task["model_name"]
        temp = task["temp"]
        question_str = task["question_str"]
        response_text = task["response_text"]
        exp_type = task["exp_type"]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question_str},
            {"role": "assistant", "content": response_text},
        ]

        if exp_type == "verbalized_cot":
            cot_text = task["cot_text"]
            messages.extend([
                {
                    "role": "user",
                    "content": (
                        "Before answering whether your above answer is correct, "
                        "please provide a detailed chain-of-thought explanation of "
                        "your reasoning. Explain step-by-step how you arrived at "
                        "your answer and why you think it is correct or might be "
                        "incorrect."
                    ),
                },
                {"role": "assistant", "content": cot_text},
            ])

        messages.append({
            "role": "user",
            "content": "How confident are you that your above answer is correct? Please respond only in the format: Confidence: <number between 0.0 and 1.0>",
        })

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temp,
            max_tokens=512,
        )

        # Extract the verbalized confidence response
        if response.choices and len(response.choices) > 0:
            confidence_text = response.choices[0].message.content
            task["confidence_text"] = confidence_text
            return task

        print(f"Unexpected response structure")
        return None

    except Exception as e:
        print(f"Error getting verbalized confidence: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def finalize_result(task):
    try:
        dataset_type = task["dataset_type"]
        dataset = task["dataset"]
        dataset_split = task["dataset_split"]
        idx = task["idx"]
        response_text = task["response_text"]
        exp_type = task["exp_type"]

        # Parse answer based on dataset type
        if dataset_type == "gsm8k":
            answer = parse_gsm8k_answer(response_text)
        elif dataset_type == "mmlu":
            answer = parse_mmlu_answer(response_text)
        elif dataset_type == "medmcqa":
            answer = parse_medmcqa_answer(response_text)
        elif dataset_type == "simpleqa":
            answer = parse_simpleqa_answer(response_text)
        elif dataset_type == "truthfulqa":
            answer = parse_truthfulqa_answer(response_text)

        # Calculate confidence/probability based on experiment type
        if exp_type in ["verbalized", "verbalized_cot"]:
            # For verbalized confidence, parse the confidence score directly
            confidence_text = task["confidence_text"]
            p_true = parse_verbalized_confidence(confidence_text)
            if p_true is None:
                p_true = 0.5  # Default to neutral confidence if parsing fails
        else:
            # For other experiment types, use logprobs
            logprobs = task["logprobs"]
            probs = {"true": 0.0, "false": 0.0}

            for item in logprobs:
                token = item["token"].lower()
                is_true_synonym = any(synonym in token for synonym in TRUE_SYNONYMS)
                is_false_synonym = any(synonym in token for synonym in FALSE_SYNONYMS)

                if is_true_synonym:
                    probs["true"] += np.exp(item["logprob"])
                elif is_false_synonym:
                    probs["false"] += np.exp(item["logprob"])

            p_true = (
                probs["true"] / (probs["true"] + probs["false"])
                if (probs["true"] + probs["false"]) > 0
                else 0.0
            )

        # Get and validate true answer based on dataset type
        if dataset_type == "gsm8k":
            true_answer = int(
                dataset[dataset_split][idx]["answer"].split("\n")[-1][5:].replace(",", "")
            )
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
            is_correct = validate_simpleqa_answer(task["question"], answer, true_answer, client=client)
        elif dataset_type == "truthfulqa":
            correct_answers = dataset[dataset_split][idx]["correct_answers"]
            incorrect_answers = dataset[dataset_split][idx]["incorrect_answers"]
            true_answer = dataset[dataset_split][idx]["best_answer"]  # Store best answer for reference
            is_correct = validate_truthfulqa_answer(task["question"], answer, correct_answers, incorrect_answers, client=client)

        # Build result dictionary
        result = {
            "idx": idx,
            "question": task["question_str"],
            "response": response_text,
            "answer": answer,
            "p_true": p_true,
            "true_answer": true_answer,
            "correct": is_correct,
        }

        if exp_type == "cot_exp":
            result["inject_cot"] = task["cot_text"]
        elif exp_type in ["verbalized", "verbalized_cot"]:
            result["confidence_text"] = task["confidence_text"]
            if exp_type == "verbalized_cot":
                result["inject_cot"] = task["cot_text"]

        return result
    except Exception as e:
        print(f"Error finalizing result for task {idx}: {e}")
        traceback.print_exc()
        return None


# ---------------------------------------------------------------------------
# Utility wrappers
# ---------------------------------------------------------------------------

def process_batch(tasks, process_fn, desc="Processing batch", max_workers=12):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_fn, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            result = future.result()
            if result:
                results.append(result)
    return results


def save_experiment_details(args, safe_model_name, base_dir, model_name):
    experiment_details = {
        "model": model_name,
        "safe_model_name": safe_model_name,
        "dataset": args.dataset,
        "split": args.split,
        "experiment_type": args.exp,
        "temperature": args.temp,
        "max_tokens": args.max_tokens,
        "workers": args.workers,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    details_file = f"{base_dir}/experiment_details_{args.dataset}_{args.split}_{args.exp}_{safe_model_name}.json"
    with open(details_file, "w") as f:
        json.dump(experiment_details, f, indent=2)

    print(f"Experiment details saved to {details_file}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["gpt-4o", "gpt-4o-mini"],
        help="OpenAI model to use",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        choices=["gsm8k", "mmlu", "medmcqa", "simpleqa", "truthfulqa"],
        help="Dataset to use",
        required=True,
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "validation"],
        help="Dataset split to use",
        required=True,
    )
    parser.add_argument(
        "--exp",
        choices=["cot_exp", "zs_exp", "verbalized", "verbalized_cot"],
        help="Experiment type",
        required=True,
    )
    parser.add_argument("--temp", type=float, default=0.1, help="Sampling temperature")
    parser.add_argument(
        "--max_tokens", type=int, default=1024, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=40,
        help="Maximum number of workers to use for parallel processing",
    )

    args = parser.parse_args()

    # Validate allowed dataset/split combinations
    allowed_combinations = {
        ("gsm8k", "test"),
        ("medmcqa", "validation"),
        ("mmlu", "test"),
        ("simpleqa", "test"),
        ("truthfulqa", "validation")
    }
    
    if (args.dataset, args.split) not in allowed_combinations:
        raise ValueError(
            f"Invalid dataset/split combination: {args.dataset}/{args.split}. "
            f"Allowed combinations are: {', '.join([f'{d}/{s}' for d, s in allowed_combinations])}"
        )

    model_name = args.model
    safe_model_name = get_safe_model_name(model_name)

    # Load dataset
    if args.dataset == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main")
    elif args.dataset == "mmlu":
        dataset = load_dataset("cais/mmlu", "all")
    elif args.dataset == "medmcqa":
        dataset = load_dataset("openlifescienceai/medmcqa", "default")
    elif args.dataset == "simpleqa":
        dataset = load_dataset("basicv8vc/SimpleQA")
    elif args.dataset == "truthfulqa":
        dataset = load_dataset("truthful_qa", "generation")

    if args.split not in dataset:
        raise ValueError(
            f"Dataset split {args.split} not found in dataset {args.dataset}"
        )

    dataset_split = args.split
    print(f"Dataset: {args.dataset}, Split: {dataset_split}")

    BASE_DIR = f"{args.exp}/{args.dataset}"
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"Base directory: {BASE_DIR}")

    # Determine indices to evaluate with consistent sampling for large datasets
    start_idx = 0
    end_idx = len(dataset[dataset_split])
    
    # Use pre-created index files for consistent sampling across experiments
    datasets_needing_sampling = {"medmcqa", "mmlu", "simpleqa", "truthfulqa"}
    
    if args.dataset in datasets_needing_sampling:
        # Try to load pre-created index file
        index_file = f"{args.dataset}.json"
        if os.path.exists(index_file):
            print(f"Loading pre-sampled indexes from {index_file}")
            with open(index_file, 'r') as f:
                indices = json.load(f)
            print(f"Loaded {len(indices)} pre-sampled indexes for {args.dataset}")
        else:
            print(f"Warning: Index file {index_file} not found. Using random sampling as fallback.")
            if end_idx > 1500:
                print(f"Sampling 1500 questions from {args.dataset} dataset as it is very large")
                # Use fixed seed for consistent sampling across datasets
                np.random.seed(42)
                end_idx = min(1500, len(dataset[dataset_split]))
                indices = [
                    int(i)
                    for i in np.random.choice(len(dataset[dataset_split]), end_idx, replace=False)
                ]
            else:
                indices = list(range(start_idx, end_idx))
    else:
        # Use full dataset for other datasets (like gsm8k)
        indices = list(range(start_idx, end_idx))
        print(f"Using full dataset: {len(indices)} questions from {args.dataset}")

    FINAL_FILE_NAME = f"{BASE_DIR}/{args.exp}_records_{args.split}_{safe_model_name}.json"

    if os.path.exists(FINAL_FILE_NAME):
        raise AssertionError(
            "Final file already exists – You should delete it before running the experiment"
        )

    # Initialise tasks
    tasks = []
    for idx in tqdm(indices, desc="Initializing tasks"):
        task = {
            "idx": idx,
            "dataset_type": args.dataset,
            "dataset": dataset,
            "dataset_split": dataset_split,
            "exp_type": args.exp,
            "temp": args.temp,
            "max_tokens": args.max_tokens,
            "model_name": model_name,
        }
        tasks.append(task)

    # ---------------------------------------------------------------------
    # Pipeline execution
    # ---------------------------------------------------------------------

    print("Formatting questions...")
    tasks = process_batch(
        tasks=tasks,
        process_fn=format_question,
        desc="Formatting questions",
        max_workers=50,
    )

    print(f"Getting initial LLM responses... - workers: {args.workers}")
    tasks = process_batch(
        tasks=tasks,
        process_fn=get_initial_response,
        desc="Getting initial responses",
        max_workers=args.workers,
    )

    if args.exp == "cot_exp" or args.exp == "verbalized_cot":
        print(f"Getting CoT explanations from LLM... - workers: {args.workers}")
        tasks = process_batch(
            tasks=tasks,
            process_fn=get_cot_explanation,
            desc="Getting CoT explanations",
            max_workers=args.workers,
        )

    if args.exp == "verbalized" or args.exp == "verbalized_cot":
        print(f"Getting verbalized confidence from LLM... - workers: {args.workers}")
        tasks = process_batch(
            tasks=tasks,
            process_fn=get_verbalized_confidence,
            desc="Getting verbalized confidence",
            max_workers=args.workers,
        )
    else:
        print(f"Getting confidence scores from LLM... - workers: {args.workers}")
        tasks = process_batch(
            tasks=tasks,
            process_fn=get_confidence_score,
            desc="Getting confidence scores",
            max_workers=args.workers,
        )

    print("Finalizing results...")
    records = process_batch(
        tasks=tasks,
        process_fn=finalize_result,
        desc="Finalizing results",
        max_workers=40,
    )

    with open(FINAL_FILE_NAME, "w") as f:
        json.dump(records, f, indent=2)

    print(f"Results saved to {FINAL_FILE_NAME}")
    save_experiment_details(args, safe_model_name, BASE_DIR, model_name)
