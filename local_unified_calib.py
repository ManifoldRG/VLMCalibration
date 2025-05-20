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
from openai import OpenAI
from dotenv import load_dotenv
import subprocess

from utils import *


load_dotenv()

# OpenAI configuration for SimpleQA grading
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def send_request(payload, url=None):
    if url is None:
        raise ValueError("URL is not set")
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError as e:
        print(f"Request error: {e}")
        print("Connection error occurred. Continuing with available results.")
        exit(1)
    except Exception as e:
        print(f"Request error: {e}")
        return None

def get_model_info(base_url):
    """Fetch model information from the OpenAI-compatible Models API."""
    try:
        models_url = f"{base_url}/models"
        response = requests.get(models_url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching model info: {e}")
        return None

def get_safe_model_name(model_name: str) -> str:
    # Replace slashes and other special chars with underscores
    model_name = model_name.split("/")[-1]
    safe_name = re.sub(r'[/\\:]', '_', model_name)
    # Remove any other non-alphanumeric chars except underscores and dashes
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '', safe_name)
    return safe_name

    
TRUE_SYNONYMS = [
    "true",
    "correct", 
    "truth",
    "yes",
    "right", 
    "verdade"
]

FALSE_SYNONYMS = [
    "false",
    "incorrect", 
    "wrong",
    "fake",
    "no",
    "not",
    "none"
]


def format_question(task):
    try:
        idx = task["idx"]
        dataset_type = task["dataset_type"]
        dataset = task["dataset"]
        dataset_split = task["dataset_split"]
        
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
        
        task["question"] = question
        task["question_str"] = question_str
        return task
    except Exception as e:
        print(f"Error formatting question {idx}: {e}")
        traceback.print_exc()
        return None


def get_initial_response(task):
    try:
        url = task["url"]
        temp = task["temp"]
        max_tokens = task["max_tokens"]
        question_str = task["question_str"]
        
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question_str},
            ],
            "temperature": temp,
            "max_tokens": max_tokens,
        }
        
        response = send_request(payload, url)
        if not response:
            return None
            
        response_text = response["choices"][0]["message"]["content"]
        task["response_text"] = response_text
        return task
    except Exception as e:
        print(f"Error getting initial response for task: {e}")
        traceback.print_exc()
        return None


def get_cot_explanation(task):
    try:
        if task["exp_type"] != "cot_exp":
            return task
            
        url = task["url"]
        temp = task["temp"]
        max_tokens = task["max_tokens"]
        question_str = task["question_str"]
        response_text = task["response_text"]
        
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question_str},
                {"role": "assistant", "content": response_text},
                {
                    "role": "user",
                    "content": "Before answering whether your above answer is correct, please provide a detailed chain-of-thought explanation of your reasoning. Explain step-by-step how you arrived at your answer and why you think it is correct or might be incorrect.",
                },
            ],
            "temperature": temp,
            "max_tokens": max_tokens,
        }
        
        response = send_request(payload, url)
        if not response:
            return None
            
        cot_text = response["choices"][0]["message"]["content"]
        task["cot_text"] = cot_text
        return task
    except Exception as e:
        print(f"Error getting CoT explanation: {e}")
        traceback.print_exc()
        return None


def get_confidence_score(task):
    try:
        url = task["url"]
        temp = task["temp"]
        # max_tokens is not used for logprobs step - only ask for 1 token in this type of experiment
        question_str = task["question_str"]
        response_text = task["response_text"]
        exp_type = task["exp_type"]
        
        payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question_str},
                {"role": "assistant", "content": response_text},
            ],
        }
        
        if exp_type == "cot_exp":
            cot_text = task["cot_text"]
            payload["messages"].extend([
                {
                    "role": "user",
                    "content": "Before answering whether your above answer is correct, please provide a detailed chain-of-thought explanation of your reasoning. Explain step-by-step how you arrived at your answer and why you think it is correct or might be incorrect.",
                },
                {"role": "assistant", "content": cot_text},
            ])
        
        payload["messages"].append({
            "role": "user",
            "content": "Is the above answer correct? Answer only with the single word 'true' or 'false'.",
        })
        
        payload["temperature"] = temp
        payload["max_tokens"] = 1
        payload["n_probs"] = 25
        
        response = send_request(payload, url)
        if not response:
            return None
            
        logprobs = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        task["logprobs"] = logprobs
        return task
    except Exception as e:
        print(f"Error getting confidence score: {e}")
        traceback.print_exc()
        return None


def finalize_result(task):
    try:
        dataset_type = task["dataset_type"]
        dataset = task["dataset"]
        dataset_split = task["dataset_split"]
        idx = task["idx"]
        response_text = task["response_text"]
        logprobs = task["logprobs"]
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
        
        # Calculate probability of true/false
        probs = {"true": 0.0, "false": 0.0}
        
        for item in logprobs:
            token = item["token"].lower()
            # Check if the token is a synonym for true or false
            is_true_synonym = any(synonym in token for synonym in TRUE_SYNONYMS)
            is_false_synonym = any(synonym in token for synonym in FALSE_SYNONYMS)

            if is_true_synonym:
                probs["true"] += np.exp(item["logprob"])
            elif is_false_synonym:
                probs["false"] += np.exp(item["logprob"])

        p_true = probs["true"] / (probs["true"] + probs["false"]) if (probs["true"] + probs["false"]) > 0 else 0.0
        
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
            client = task.get("client")
            is_correct = validate_simpleqa_answer(task["question"], answer, true_answer, client=client)
        
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
            
        return result
    except Exception as e:
        print(f"Error finalizing result for task {idx}: {e}")
        traceback.print_exc()
        return None

# Process tasks in batch using ThreadPoolExecutor
def process_batch(tasks, process_fn, desc="Processing batch", max_workers=12):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_fn, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            result = future.result()
            if result:
                results.append(result)
    return results

# Save experiment details to a separate file
def save_experiment_details(args, safe_model_name, base_dir, model_name, model_info=None):
    experiment_details = {
        "model": model_name,
        "safe_model_name": safe_model_name,
        "dataset": args.dataset,
        "split": args.split,
        "experiment_type": args.exp,
        "temperature": args.temp,
        "max_tokens": args.max_tokens,
        "port": args.port,
        "workers": args.workers,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add model information if available
    if model_info:
        experiment_details["model_info"] = model_info
    
    details_file = f"{base_dir}/experiment_details_{safe_model_name}.json"
    with open(details_file, 'w') as f:
        json.dump(experiment_details, f, indent=2)
    
    print(f"Experiment details saved to {details_file}")

if __name__ == "__main__":
    import argparse
    '''
    python local_unified_calib.py --dataset gsm8k --split test --exp cot_exp
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["gsm8k", "mmlu", "medmcqa", "simpleqa"], help="Dataset to use", required=True)
    parser.add_argument("--split", choices=["train", "test", "validation"], help="Dataset split to use", required=True)
    parser.add_argument("--exp", choices=["cot_exp", "zs_exp", "few_shot"], help="Experiment type", required=True)
    parser.add_argument("--port", type=int, default=8080, help="Port to use for local model")
    parser.add_argument("--temp", type=float, default=0.1, help="Temperature to use for local model")
    parser.add_argument("--max_tokens", type=int, default=1024, help="Maximum tokens to use for local model")
    parser.add_argument("--workers", type=int, default=None, help="Maximum number of workers to use")

    args = parser.parse_args()

    # Automatically determine the number of workers from the running llama-server process
    if args.workers is None:
        try:
            result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
            output = result.stdout
            server_lines = [line for line in output.split('\n') if 'llama-server' in line]
            
            # Set default in case we can't find the setting
            args.workers = 12
            
            # Find -np value in the command line args of llama-server
            if server_lines:
                for line in server_lines:
                    np_match = re.search(r'-np\s+(\d+)', line)
                    if np_match:
                        np_value = int(np_match.group(1))
                        # Use the same number of workers as the server's -np value
                        args.workers = np_value
                        print(f"Automatically set workers to {args.workers} based on llama-server -np parameter")
                        break
        except Exception as e:
            print(f"Error detecting llama-server workers: {e}")
            print("Using default workers=12")
            args.workers = 12

    # Base URL for API endpoints
    base_url = f"http://localhost:{args.port}/v1"
    url = f"{base_url}/chat/completions"
    
    # Get model information from the API
    model_info = get_model_info(base_url)
    if not model_info or "data" not in model_info or len(model_info["data"]) == 0:
        print("Warning: Could not retrieve model information from API")
        model_name = "unknown_model"
    else:
        model_name = model_info["data"][0]["id"]
        print(f"Using model from API: {model_name}")
    
    safe_model_name = get_safe_model_name(model_name)

    assert not (args.dataset == "medmcqa" and args.split == "test"), "MedMCQA test split does not contain answers - cannot calibrate - use validation split instead"
    
    # Load the dataset
    if args.dataset == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main")
    elif args.dataset == "mmlu":
        dataset = load_dataset("cais/mmlu", "all")
    elif args.dataset == "medmcqa":
        dataset = load_dataset("openlifescienceai/medmcqa", "default")
    elif args.dataset == "simpleqa":
        dataset = load_dataset("basicv8vc/SimpleQA")

    if args.split not in dataset:
        raise ValueError(f"Dataset split {args.split} not found in dataset {args.dataset}")
 
    dataset_split = args.split
    print(f"Dataset: {args.dataset}, Split: {dataset_split}")
    
    BASE_DIR = f"{args.exp}/{args.dataset}"
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"Base directory: {BASE_DIR}")

    # Initialize indices for the data
    start_idx = 0
    end_idx = len(dataset[dataset_split])

    if end_idx > 5000:
        print("Sampling 5000 questions from dataset as it is very large")
        np.random.seed(42)
        end_idx = min(5000, len(dataset[dataset_split]))
        indices = [int(i) for i in np.random.choice(len(dataset[dataset_split]), end_idx, replace=False)]
    else:
        indices = list(range(start_idx, end_idx))

    FINAL_FILE_NAME = f"{BASE_DIR}/{args.exp}_records_{args.split}_{safe_model_name}.json"

    assert not os.path.exists(FINAL_FILE_NAME), "Final file already exists - You should delete it before running the experiment"
    

    # Initialize tasks
    tasks = []
    for idx in tqdm(indices, desc="Initializing tasks"):
        task = {
            "idx": idx,
            "dataset_type": args.dataset,
            "dataset": dataset,
            "dataset_split": dataset_split,
            "exp_type": args.exp,
            "url": url,
            "temp": args.temp,
            "max_tokens": args.max_tokens,
        }
        if args.dataset == "simpleqa":
            task["client"] = openai_client
        tasks.append(task)

    # Format all questions according to dataset type
    print("Formatting questions...")
    tasks = process_batch(tasks=tasks, process_fn=format_question, desc="Formatting questions", max_workers=50)
    
    # Get initial LLM responses for all formatted questions
    print("Getting initial LLM responses...")
    tasks = process_batch(tasks=tasks, process_fn=get_initial_response, desc="Getting initial responses", max_workers=args.workers)
    
    # Get CoT explanations for all responses if needed
    if args.exp == "cot_exp":
        print("Getting CoT explanations from LLM...")
        tasks = process_batch(tasks=tasks, process_fn=get_cot_explanation, desc="Getting CoT explanations", max_workers=args.workers)
    
    # Get confidence scores for all responses
    print("Getting confidence scores from LLM...")
    tasks = process_batch(tasks=tasks, process_fn=get_confidence_score, desc="Getting confidence scores", max_workers=args.workers)
    
    # Finalize results by parsing answers and validating
    print("Finalizing results...")
    records = process_batch(tasks=tasks, process_fn=finalize_result, desc="Finalizing results", max_workers=40)

    # Save results as JSON
    with open(FINAL_FILE_NAME, 'w') as f:
        json.dump(records, f, indent=2)
    
    print(f"Results saved to {FINAL_FILE_NAME}")
    save_experiment_details(args, safe_model_name, BASE_DIR, model_name, model_info)