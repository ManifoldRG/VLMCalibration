from llama_cpp import Llama
from llama_cpp.llama_chat_format import format_llama3
import numpy as np
from datasets import load_dataset
import re
import pandas as pd
from tqdm import tqdm

def get_answer(answer_text):
    return int(answer_text.split('\n')[-1][5:].replace(",", ""))

llm = Llama(
      model_path="/models/Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf",
      n_gpu_layers=-1,
      logits_all=True,
      verbose=False
)

# llm = Llama.from_pretrained(
#       repo_id="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
#       filename="*Q8_0.gguf",
#       n_gpu_layers=-1,
#       logits_all=True,
#       verbose=False
# )

dataset = load_dataset("openai/gsm8k", "main")
records = []
dataset_split = 'train'

for i in tqdm(range(len(dataset[dataset_split]))):
    try:
        question = dataset[dataset_split][i]['question']

        messages = [
            {'role': 'user', 'content': f"Given the following problem, reason and give a final answer to the problem. {question} Let's think step by step. At the end, you MUST write the answer as an integer after '####'."},
        ]

        response = llm(
            format_llama3(llama=llm, messages=messages).prompt, # type: ignore
            max_tokens=256,
        )['choices'][0]['text']
        answer = int(re.findall(r'\d+', response)[-1])

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
        # print(response)
        # print(p_true)
        # print()
        true_answer = get_answer(dataset[dataset_split][i]['answer'])
        records.append({
            'question': question,
            'response': response,
            'p_true': p_true,
            'true_answer': true_answer,
            'answer': answer,
            'correct': abs(answer - true_answer) < 0.0001
        })
        if (i > 0) and (i % 100):
            pd.DataFrame.from_records(records).to_csv('records_.csv')
    except:
        continue
pd.DataFrame.from_records(records).to_csv('records_train_.csv')