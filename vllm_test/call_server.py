from openai import OpenAI
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
import time
import re
import numpy as np

def parse_answer(output: str) -> int:
    # Look for "The final answer is X" pattern
    final_answer_match = re.search(r"The final answer is \$?(\d+,?\d*)", output)
    if final_answer_match:
        return int(final_answer_match.group(1).replace(",", ""))
    # Fallback: find all numbers and take the last one
    matches = re.findall(r"\$?(\d+,?\d*)", output)
    return int(matches[-1].replace(",", "")) if matches else None

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="token-abc123",
)

dataset = load_dataset("openai/gsm8k", "main")
dataset_split = "train"

def process_question(idx):
    question = dataset[dataset_split][idx]["question"]
    prompt = f'Given the following problem, reason and give a final answer to the problem.\nProblem: {question}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.'
    
    completion = client.chat.completions.create(
        model="./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        messages=[
            {"role": "user", "content": prompt}
        ],
        logprobs=False
    )
    response_text = completion.choices[0].message.content
    answer = parse_answer(response_text)
    final_messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response_text},
            {
                "role": "user",
                "content": f"Is the above answer correct? Answer only with the single word 'true' or 'false'. The correct answer is {answer}.",
            },
        ]
    completion = client.chat.completions.create(
        model="./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        messages=final_messages,
        logprobs=True,
        top_logprobs=20
    )
    return completion

start_time = time.time()
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = []
    for idx in range(100):
        futures.append(executor.submit(process_question, idx))

    results = []
    for future in tqdm(as_completed(futures), total=100, desc="Processing questions"):
        result = future.result()
        logprobs = result.choices[0].logprobs.content[0].top_logprobs
        probs = {"true": 0.0, "false": 0.0}
        
        for item in logprobs:
            token = item.token.lower()
            if token in probs:
                probs[token] += np.exp(item.logprob)  # Convert logprob to probability
        
        p_true = probs['true'] / (probs['true'] + probs['false'])
        p_false = probs['false'] / (probs['true'] + probs['false'])
        print(f"True probability: {p_true}, False probability: {p_false}")

end_time = time.time()
print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")