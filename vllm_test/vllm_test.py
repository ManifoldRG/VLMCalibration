import os
os.environ["HF_HOME"] = "./models"

from dotenv import load_dotenv
load_dotenv()
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")


from vllm import LLM, SamplingParams
from datasets import load_dataset
import time

dataset = load_dataset("openai/gsm8k", "main")
dataset_split = "train"

llm = LLM(
    model="./models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
    tokenizer="meta-llama/Llama-3.1-8B-Instruct",
    trust_remote_code=True,
    max_model_len=4096
)

tokenizer = llm.get_tokenizer()

conversations = []
for idx in range(10):
    question = dataset[dataset_split][idx]["question"]
    prompt = f'Given the following problem, reason and give a final answer to the problem.\nProblem: {question}\nYour response should end with "The final answer is [answer]" where [answer] is the response to the problem.'
    conversation = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': prompt}],
        tokenize=False
    )
    conversations.append(conversation)

sampling_params = SamplingParams(
    temperature=0.5,
    top_p=0.9,
    max_tokens=1024,
    stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
)

start_time = time.time()
outputs = llm.generate(conversations, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")