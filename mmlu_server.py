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

dataset = load_dataset("cais/mmlu", "all")
records = []
dataset_split = "test"


def parse_answer(output: str) -> str:
    # Look for "The final answer is X" pattern where X is a letter A-D
    final_answer_match = re.search(r"The final answer is ([A-D])", output)
    if final_answer_match:
        return final_answer_match.group(1)
    return None

def get_answer(answer_text):
    return answer_text


def send_request(payload):
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Request error: {e}")
        return None


def process_single_question(idx):
    try:
        question = dataset[dataset_split][idx]["question"]
        choices_list = dataset[dataset_split][idx]["choices"]
        subject = dataset[dataset_split][idx]["subject"].replace("_", " ")
        formatted_choices = "\n".join([f"{chr(65 + i)}: {choice}" for i, choice in enumerate(choices_list)])
        question_str = (
            f"The following is a multiple choice question (with choices) about {subject}.\n"
            f"Reason and give a final answer to the question. Your response should end with 'The final answer is [answer]' where [answer] is solely the letter corresponding to the correct choice.\n"
            f"Question: {question}\n"
            f"Choices:\n{formatted_choices}"
        )

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

        # Second request for confidence check
        second_payload = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question_str},
                {"role": "assistant", "content": response_text},
                {
                    "role": "user",
                    "content": "Is the above answer correct? Answer only with the single word 'true' or 'false'.",
                },
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
            token = item["token"].lower()
            if token in probs:
                probs[token] += np.exp(item["logprob"])

        p_true = probs["true"] / (probs["true"] + probs["false"])
        true_answer_idx = get_answer(dataset[dataset_split][idx]["answer"])
        choices_str = ["A", "B", "C", "D"]
        true_answer = choices_str[true_answer_idx]

        return {
            "question": question,
            "response": response_text,
            "answer": answer,
            "p_true": p_true,
            "true_answer": true_answer,
            "correct": answer == true_answer if answer is not None else None,
        }
    except Exception as e:
        print(f"Error processing question {idx}: {e}")
        return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", choices=["llama", "qwen", "gemma"], help="Name of model to use")
    parser.add_argument("--dataset_split", choices=["train", "test"], help="Dataset split to use")
    args = parser.parse_args()
    assert dataset_split == args.dataset_split, f"Dataset split {dataset_split} does not match {args.dataset_split}"

    start_idx = 0  # Starting from beginning of dataset
    end_idx = len(dataset[dataset_split])

    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = []
        for idx in range(start_idx, end_idx):
            futures.append(executor.submit(process_single_question, idx))

        for idx, future in enumerate(tqdm(as_completed(futures), total=end_idx - start_idx, desc="Processing dataset")):
            result = future.result()
            if result:
                records.append(result)

            if (len(records) > 0) and (len(records) % 100 == 0):
                pd.DataFrame.from_records(records).to_csv(f"records_{args.dataset_split}_full_mid_{args.model_name}.csv")

    # Final save
    pd.DataFrame.from_records(records).to_csv(f"records_{args.dataset_split}_full_{args.model_name}.csv")
