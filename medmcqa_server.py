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

# Server configuration
url = "http://localhost:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}

def parse_answer(output: str) -> str:
    # Look for answer letter in response
    answer_match = re.search(r"The final answer is ([A-D])", output, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    # Fallback: find all letters and take the last one
    matches = re.findall(r"[A-D]", output, re.IGNORECASE)
    return matches[-1].upper() if matches else None

def send_request(payload):
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Request error: {e}")
        return None

def process_single_question(dataset, dataset_split, idx):
    try:
        question = dataset[dataset_split][idx]["question"]
        options = [
            dataset[dataset_split][idx]["opa"],
            dataset[dataset_split][idx]["opb"], 
            dataset[dataset_split][idx]["opc"],
            dataset[dataset_split][idx]["opd"]
        ]
        options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
        question_str = f'Given the following medical question and options, reason and select the correct answer letter (A-D).\nQuestion: {question}\nOptions:\n{options_str}\nYour response should end with "The final answer is [letter]" where [letter] is A, B, C or D.'

        # First request for getting the answer
        first_payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": question_str},
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
        }

        first_response = send_request(first_payload)
        if not first_response:
            return None

        response_text = first_response["choices"][0]["message"]["content"]
        answer = parse_answer(response_text)


        # Second request for confidence check
        second_payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": question_str},
                {"role": "assistant", "content": response_text},
                {"role": "user", "content": "Is the above answer correct? Answer only with the single word 'true' or 'false'."},
            ],
            "temperature": 0.7,
            "max_tokens": 1,
            "n_probs": 25,
        }

        second_response = send_request(second_payload)
        if not second_response:
            return None

        logprobs = second_response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]

        probs = {"true": 0.0, "false": 0.0}
        for item in logprobs:
            # print(item)
            token = item["token"].lower()
            if any(key in token for key in ["true", "false", "correct", "incorrect", "wrong",
                                          "truth", "yes", "right", "verdade",  # True variations
                                          "fake", "no", "not", "none"]):      # False variations
                if ("true" in token or "correct" in token or "truth" in token or 
                    "yes" in token or "right" in token or "verdade" in token):
                    # print(f"Adding for token: {token} (mapped to true)")
                    actual_key = "true"
                elif ("false" in token or "incorrect" in token or "wrong" in token or 
                      "fake" in token or "no" in token or "not" in token or "none" in token):
                    # print(f"Adding for token: {token} (mapped to false)")
                    actual_key = "false"
                probs[actual_key] += np.exp(item["logprob"])

        p_true = probs["true"] / (probs["true"] + probs["false"])
        # print(f"True probability: {p_true}, False probability: {1 - p_true}")
        true_answer = chr(65 + dataset[dataset_split][idx]["cop"])  # Convert 1-4 to A-D

        return {
            "question": question,
            "options": options,
            "response": response_text,
            "answer": answer,
            "p_true": p_true,
            "true_answer": true_answer,
            "correct": answer == true_answer if answer is not None else None
        }
    except Exception as e:
        print(f"Error processing question {idx}: {e}")
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", choices=["llama", "qwen", "gemma"], help="Name of model to use")
    parser.add_argument("dataset_split", choices=["train", "validation", "test"], help="Dataset split to use")
    parser.add_argument("exp_type", choices=["cot_exp", "zs_exp"], help="Experiment type")
    args = parser.parse_args()
    
    dataset = load_dataset("openlifescienceai/medmcqa", "default")
    dataset_split = args.dataset_split
    print(f"Dataset split: {dataset_split}")
    BASE_DIR = f"{args.exp_type}/medmcqa"
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"Base directory: {BASE_DIR}")


    records = []
    start_idx = 0
    # end_idx = len(dataset[dataset_split])
    end_idx = 4000
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = []
        for idx in range(start_idx, end_idx):
            futures.append(executor.submit(process_single_question, dataset, dataset_split, idx))

        for idx, future in enumerate(tqdm(as_completed(futures), total=end_idx - start_idx, desc="Processing dataset")):
            result = future.result()
            if result:
                records.append(result)

            if (len(records) > 0) and (len(records) % 50 == 0):
                pd.DataFrame.from_records(records).to_csv(f"{BASE_DIR}/{args.exp_type}_records_{args.dataset_split}_partial_{args.model_name}.csv")

    # Final save
    pd.DataFrame.from_records(records).to_csv(f"{BASE_DIR}/{args.exp_type}_records_{args.dataset_split}_full_{args.model_name}.csv")
