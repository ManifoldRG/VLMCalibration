import numpy as np
from datasets import load_dataset
import re
import pandas as pd
from tqdm.auto import tqdm
from llama_cpp import Llama
from llama_cpp.llama_chat_format import format_llama3


dataset = load_dataset("openai/gsm8k", "main")
records = []
dataset_split = 'train'

def parse_answer(output: str) -> int:
    # Look for "The final answer is X" pattern
    final_answer_match = re.search(r'The final answer is (\d+)', output)
    if final_answer_match:
        return int(final_answer_match.group(1))
    # print(f"Fallback: {output}")
    # Fallback: find all numbers and take the last one
    matches = re.findall(r'-?\d*\.?\d+', output)
    # print(int(float(matches[-1])))
    return int(float(matches[-1])) if matches else None

def get_answer(answer_text):
    return int(answer_text.split('\n')[-1][5:].replace(",", ""))




if __name__ == "__main__":
    llm = Llama(
        model_path="./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        n_gpu_layers=-1,
        logits_all=True,
        verbose=False,
        n_ctx=8192,
        n_batch=8192,
        n_ubatch=8192,
        n_threads=16,
        n_threads_batch=16
    )
    
    try:
        for idx in tqdm(range(2511, len(dataset[dataset_split])), total=len(dataset[dataset_split])-2511, desc="Processing dataset"):
            question = dataset[dataset_split][idx]['question']
            question_str = f"Given the following problem, reason and give a final answer to the problem.\nProblem: {question}\nYour response should end with \"The final answer is [answer]\" where [answer] is the response to the problem."
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question_str}
            ]

            response = llm(
                        format_llama3(llama=llm, messages=messages).prompt,  # type: ignore
                        max_tokens=1024,
                    )['choices'][0]['text']
            answer = parse_answer(response)

            messages.extend([
                {'role': 'assistant', 'content': response},
                {'role': 'user', 'content': 'Is the above answer correct? Answer only with the single word \'true\' or \'false\'.'},
            ])

            logprobs = llm(
                format_llama3(llama=llm, messages=messages).prompt, # type: ignore
                logprobs=25,
                max_tokens=1,
            )['choices'][0]['logprobs']['top_logprobs'][0]

            probs = {'true': 0., 'false': 0.}

            for a, b in {a: np.exp(b) for a, b in logprobs.items()}.items(): # type: ignore
                if a.lower() in probs:
                    probs[a.lower()] += b

            p_true = probs['true'] / (probs['true'] + probs['false'])
            
            true_answer = get_answer(dataset[dataset_split][idx]['answer'])
            records.append({
                'question': question,
                'response': response,
                'answer': answer,
                'p_true': p_true,
                'true_answer': true_answer,
                'correct': (abs(answer - true_answer) < 0.0001) 
                            if (answer is not None and isinstance(true_answer, (int, float))) 
                            else None
            })
            if (idx > 0) and (idx % 100):
                pd.DataFrame.from_records(records).to_csv('records_train_aritra_2511.csv')
    finally:
        del llm
    pd.DataFrame.from_records(records).to_csv('records_train_aritra_2511.csv')


# Processing dataset:  22%|███████████▉                                           | 1082/4962 [1:49:48<10:04:33,  9.35s/it]Processing dataset:  22%|████████████▏                                           | 1082/4962