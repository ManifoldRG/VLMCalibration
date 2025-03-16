import numpy as np
from datasets import load_dataset
import re
import pandas as pd
from tqdm.auto import tqdm
import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import matplotlib.pyplot as plt
import seaborn as sns

# OpenAI configuration
OPENAI_API_KEY = "TEST"
client = OpenAI(api_key=OPENAI_API_KEY)

dataset = load_dataset("openai/gsm8k", "main")
records = []
dataset_split = "train"


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

        # # Request for chain of thought explanation
        # inject_cot_messages = [
        #     {"role": "system", "content": "You are a helpful assistant."},
        #     {"role": "user", "content": question_str},
        #     {"role": "assistant", "content": response_text},
        #     {
        #         "role": "user",
        #         "content": "Before answering whether your above answer is correct, please provide a detailed chain-of-thought explanation of your reasoning. Explain step-by-step how you arrived at your answer and why you think it is correct or might be incorrect.",
        #     },
        # ]

        # inject_cot_response = send_request(inject_cot_messages, model=model)
        # if not inject_cot_response:
        #     return None

        # inject_cot_text = inject_cot_response.choices[0].message.content

        # Final request for confidence check with logprobs
        # final_messages = [
        #     {"role": "system", "content": "You are a helpful assistant."},
        #     {"role": "user", "content": question_str},
        #     {"role": "assistant", "content": response_text},
        #     {
        #         "role": "user",
        #         "content": "Before answering whether your above answer is correct, please provide a detailed chain-of-thought explanation of your reasoning. Explain step-by-step how you arrived at your answer and why you think it is correct or might be incorrect.",
        #     },
        #     {"role": "assistant", "content": inject_cot_text},
        #     {
        #         "role": "user",
        #         "content": "Is the above answer correct? Answer only with the single word 'true' or 'false'.",
        #     },
        # ]
        final_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question_str},
            {"role": "assistant", "content": response_text},
            {
                "role": "user",
                "content": "Is the above answer correct? Answer only with the single word 'true' or 'false'.",
            },
        ]

        final_response = send_request(
            final_messages,
            temperature=0.7,
            max_tokens=1,
            model=model,
            logprobs=True,
            top_logprobs=20,  # Get maximum number of logprobs
        )
        if not final_response:
            return None

        # Extract logprobs for 'true' and 'false' tokens
        logprobs = final_response.choices[0].logprobs.content[0].top_logprobs
        probs = {"true": 0.0, "false": 0.0}
        
        for item in logprobs:
            # print(item)
            token = item.token.lower()
            if token in probs:
                probs[token] += np.exp(item.logprob)  # Convert logprob to probability

        # Calculate p_true as the normalized probability
        p_true = probs["true"] / (probs["true"] + probs["false"])

        true_answer = get_answer(dataset[dataset_split][idx]["answer"])

        return {
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
    except Exception as e:
        print(f"Error processing question {idx}: {e}")
        return None


def plot_calibration_metrics(df: pd.DataFrame, model_name: str):
    # Create plots directory if it doesn't exist
    import os

    os.makedirs("plots", exist_ok=True)

    # Accuracy vs Confidence plot
    plt.figure(figsize=(10, 6))
    df_grouped = df.groupby("p_true")["correct"].mean().reset_index()
    plt.scatter(df_grouped["p_true"], df_grouped["correct"])
    plt.plot([0, 1], [0, 1], "r--")  # Perfect calibration line
    plt.xlabel("Confidence (p_true)")
    plt.ylabel("Accuracy")
    plt.title(f"Calibration Plot for {model_name}")
    plt.savefig(f"plots/calibration_{model_name}.png")
    plt.close()

    # Distribution of confidences
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="p_true", bins=20)
    plt.xlabel("Confidence (p_true)")
    plt.ylabel("Count")
    plt.title(f"Distribution of Confidences for {model_name}")
    plt.savefig(f"plots/confidence_dist_{model_name}.png")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        choices=["gpt-4o", "gpt-4o-mini"],
        default="gpt-4o",
        help="Name of OpenAI model to use",
    )
    parser.add_argument(
        "--dataset_split", choices=["train", "test"], help="Dataset split to use"
    )
    args = parser.parse_args()
    assert (
        dataset_split == args.dataset_split
    ), f"Dataset split {dataset_split} does not match {args.dataset_split}"

    start_idx = 0
    end_idx = len(dataset[dataset_split])

    with ThreadPoolExecutor(
        max_workers=25
    ) as executor:  # Reduced workers due to API rate limits
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
                df = pd.DataFrame.from_records(records)
                df.to_csv(f"records_{args.dataset_split}_full_{args.model_name}.csv")
                plot_calibration_metrics(df, args.model_name)

    # Final save
    final_df = pd.DataFrame.from_records(records)
    final_df.to_csv(f"records_{args.dataset_split}_full_{args.model_name}.csv")
    plot_calibration_metrics(final_df, args.model_name)
