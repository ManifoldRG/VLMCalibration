import numpy as np
from datasets import load_dataset
import re
import pandas as pd
from tqdm.auto import tqdm
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Server configuration
url = "http://localhost:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}



def parse_answer(output: str) -> int:
    # Look for "The final answer is X" pattern
    final_answer_match = re.search(r"The final answer is \$?(\d+,?\d*)", output)
    if final_answer_match:
        return int(final_answer_match.group(1).replace(",", ""))
    # Fallback: find all numbers and take the last one
    matches = re.findall(r"\$?(\d+,?\d*)", output)
    return int(matches[-1].replace(",", "")) if matches else None

def get_answer(answer_text):
    return int(answer_text.split("\n")[-1][5:].replace(",", ""))


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
        question_str = f'Given the following problem, reason and give a final answer to the problem.\nProblem: {question}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.'

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
        answer = parse_answer(response_text)

        # inject_cot_payload = {
        #     "messages": [
        #         {"role": "system", "content": "You are a helpful assistant."},
        #         {"role": "user", "content": question_str},
        #         {"role": "assistant", "content": response_text},
        #         {"role": "user", "content": "Before answering whether your above answer is correct, please provide a detailed chain-of-thought explanation of your reasoning. Explain step-by-step how you arrived at your answer and why you think it is correct or might be incorrect."},
        #     ],
        #     "temperature": 0.7,
        #     "max_tokens": 1024,
        # }
        # inject_cot_response = send_request(inject_cot_payload)
        # if not inject_cot_response:
        #     return None

        # inject_cot_text = inject_cot_response["choices"][0]["message"]["content"]
        # Second request for confidence check
        second_payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question_str},
                {"role": "assistant", "content": response_text},
                # {"role": "user", "content": "Before answering whether your above answer is correct, please provide a detailed chain-of-thought explanation of your reasoning. Explain step-by-step how you arrived at your answer and why you think it is correct or might be incorrect."},
                # {"role": "assistant", "content": inject_cot_text},
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
            print(item)
            token = item["token"].lower()
            # Check for true/false variations and map them
            if any(key in token for key in ["true", "false", "correct", "incorrect", "wrong", 
                                          "truth", "yes", "right", "verdade",  # True variations
                                          "fake", "no", "not", "none"]):      # False variations
                # Map variations to actual keys
                if ("true" in token or "correct" in token or "truth" in token or 
                    "yes" in token or "right" in token or "verdade" in token):
                    actual_key = "true"
                elif ("false" in token or "incorrect" in token or "wrong" in token or 
                      "fake" in token or "no" in token or "not" in token or "none" in token):
                    actual_key = "false"
                print(f"Adding for token: {token} (mapped to {actual_key})")
                probs[actual_key] += np.exp(item["logprob"])

        p_true = probs["true"] / (probs["true"] + probs["false"])
        print(f"True probability: {p_true}, False probability: {1 - p_true}")
        true_answer = get_answer(dataset[dataset_split][idx]["answer"])

        return {
            "question": question,
            "response": response_text,
            "answer": answer,
            "p_true": p_true,
            # "inject_cot": inject_cot_text,
            "true_answer": true_answer,
            "correct": (
                (abs(answer - true_answer) < 0.0001)
                if (answer is not None and isinstance(true_answer, (int, float)))
                else None
            ),
        }
    except Exception as e:
        print(f"Error processing question {idx}: {e}")
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", choices=["llama", "qwen", "gemma"], help="Name of model to use")
    parser.add_argument("dataset_split", choices=["train", "test"], help="Dataset split to use")
    parser.add_argument("exp_type", choices=["cot_exp", "zs_exp"], help="Experiment type")
    args = parser.parse_args()
    dataset = load_dataset("openai/gsm8k", "main")
    dataset_split = args.dataset_split
    BASE_DIR = f"{args.exp_type}/gsm8k"

    records = []
    start_idx = 0  # Starting from beginning of dataset
    end_idx = len(dataset[dataset_split])
    with ThreadPoolExecutor(max_workers=16) as executor:
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
