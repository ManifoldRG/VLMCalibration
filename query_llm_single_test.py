import numpy as np
from datasets import load_dataset
import re
import json
import requests

# Server configuration
url = "http://localhost:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}

dataset = load_dataset("openai/gsm8k", "main")
dataset_split = "test"

def parse_answer(output: str) -> int:
    # Look for "The final answer is X" pattern
    final_answer_match = re.search(r"The final answer is \$?(\d+,?\d*)", output)
    if final_answer_match:
        return int(final_answer_match.group(1).replace(",", ""))
    # Fallback: find all numbers and take the last one
    matches = re.findall(r"\$?(\d+,?\d*)", output)
    return int(matches[-1].replace(",", "")) if matches else None

def send_request(payload):
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Request error: {e}")
        return None

if __name__ == "__main__":
    # Get first question
    question = dataset[dataset_split][0]["question"]
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
    print("First Response received!")
    response_text = first_response["choices"][0]["message"]["content"]
    
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
    logprobs = second_response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]

    print("\nTop 25 logprobs")
    for item in logprobs:
        print(f"{item['token']} - {item['logprob']}")
