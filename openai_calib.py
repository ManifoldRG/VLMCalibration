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

load_dotenv()

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def parse_answer(output: str) -> int:
    # Look for "The final answer is X" pattern
    final_answer_match = re.search(r"The final answer is \$?(\d+,?\d*)", output)
    if final_answer_match:
        return int(final_answer_match.group(1).replace(",", ""))
    # Fallback: find all numbers and take the last one
    matches = re.findall(r"\$?(\d+,?\d*)", output)
    return int(matches[-1].replace(",", "")) if matches else None


def get_answer(answer_text: str) -> int:
    return int(answer_text.split("\n")[-1][5:].replace(",", ""))


def send_request(
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    model: str = "gpt-4o",
    logprobs: bool = False,
    top_logprobs: int = None,
) -> dict:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
        )
        return response
    except Exception as e:
        print(f"Request error: {e}")
        return None


def process_single_question(idx: int, model: str = "gpt-4o"):
    try:
        question = dataset[dataset_split][idx]["question"]
        question_str = f'Given the following problem, reason and give a final answer to the problem.\nProblem: {question}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.'

        # First request for getting the answer
        first_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question_str},
        ]

        first_response = send_request(first_messages, model=model)
        if not first_response:
            return None

        response_text = first_response.choices[0].message.content
        answer = parse_answer(response_text)
        # print(f"FIRST RESPONSE: {response_text}")

        # Add chain-of-thought injection for cot_exp
        if args.exp_type == "cot_exp":
            inject_cot_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question_str},
                {"role": "assistant", "content": response_text},
                {
                    "role": "user",
                    "content": "Before answering whether your above answer is correct, please provide a detailed chain-of-thought explanation of your reasoning. Explain step-by-step how you arrived at your answer and why you think it is correct or might be incorrect.",
                },
            ]

            inject_cot_response = send_request(inject_cot_messages, model=model)
            if not inject_cot_response:
                return None
            inject_cot_text = inject_cot_response.choices[0].message.content
            # print(f"COT RESPONSE: {inject_cot_text}")
        else:
            inject_cot_text = None

        # Final request for confidence check with logprobs
        final_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question_str},
            {"role": "assistant", "content": response_text},
        ]

        # Add CoT messages if in cot_exp mode
        if args.exp_type == "cot_exp":
            final_messages.extend([
                {
                    "role": "user",
                    "content": "Before answering whether your above answer is correct, please provide a detailed chain-of-thought explanation of your reasoning. Explain step-by-step how you arrived at your answer and why you think it is correct or might be incorrect.",
                },
                {"role": "assistant", "content": inject_cot_text},
            ])

        final_messages.append({
            "role": "user",
            "content": "Is the above answer correct? Answer only with the single word 'true' or 'false'.",
        })

        final_response = send_request(
            final_messages,
            temperature=0.7,
            max_tokens=1,
            model=model,
            logprobs=True,
            top_logprobs=20,  # Get maximum number of logprobs
        )
        # print(f"SECOND RESPONSE: {final_response.choices[0].message.content}")
        if not final_response:
            return None

        # Extract logprobs for 'true' and 'false' tokens
        logprobs = final_response.choices[0].logprobs.content[0].top_logprobs
        probs = {"true": 0.0, "false": 0.0}
        
        for item in logprobs:
            token = item.token.lower()
            if any(key in token for key in ["true", "false", "correct", "incorrect", "wrong",
                                          "truth", "yes", "right", "verdade",  # True variations
                                          "fake", "no", "not", "none"]):      # False variations
                if ("true" in token or "correct" in token or "truth" in token or 
                    "yes" in token or "right" in token or "verdade" in token):
                    actual_key = "true"
                elif ("false" in token or "incorrect" in token or "wrong" in token or 
                      "fake" in token or "no" in token or "not" in token or "none" in token):
                    actual_key = "false"
                probs[actual_key] += np.exp(item.logprob)

        p_true = probs["true"] / (probs["true"] + probs["false"])
        true_answer = get_answer(dataset[dataset_split][idx]["answer"])

        result = {
            "question": question,
            "response": response_text,
            "answer": answer,
            "p_true": p_true,
            "true_answer": true_answer,
            "correct": (
                (abs(answer - true_answer) < 0.0001)
                if (answer is not None and isinstance(true_answer, (int, float)))
                else None
            ),
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
        choices=["gpt-4o", "gpt-4o-mini"],
        help="Name of OpenAI model to use",
    )
    parser.add_argument(
        "dataset_split", choices=["train", "test"], help="Dataset split to use"
    )
    parser.add_argument(
        "exp_type", choices=["cot_exp", "zs_exp"], help="Experiment type"
    )
    args = parser.parse_args()

    dataset = load_dataset("openai/gsm8k", "main")
    dataset_split = args.dataset_split
    print(f"Dataset split: {dataset_split}")
    BASE_DIR = f"{args.exp_type}/gsm8k"
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"Base directory: {BASE_DIR}")

    records = []
    start_idx = 0
    end_idx = len(dataset[dataset_split])

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = []
        for idx in range(start_idx, end_idx):
            futures.append(
                executor.submit(process_single_question, idx, args.model_name)
            )

        for idx, future in enumerate(
            tqdm(
                as_completed(futures),
                total=end_idx - start_idx,
                desc="Processing dataset",
            )
        ):
            result = future.result()
            if result:
                records.append(result)

            if (len(records) > 0) and (len(records) % 50 == 0):
                pd.DataFrame.from_records(records).to_csv(f"{BASE_DIR}/{args.exp_type}_records_{args.dataset_split}_partial_{args.model_name}.csv")

    # Final save
    final_df = pd.DataFrame.from_records(records)
    final_df.to_csv(f"{BASE_DIR}/{args.exp_type}_records_{args.dataset_split}_full_{args.model_name}.csv")

    # Delete the partial results file after final save
    partial_file = f"{BASE_DIR}/{args.exp_type}_records_{args.dataset_split}_partial_{args.model_name}.csv"
    if os.path.exists(partial_file):
        os.remove(partial_file)
