from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
import json

# Initialize LLM
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_model_len=2048,
    trust_remote_code=True,
    dtype="bfloat16",
    gpu_memory_utilization=0.90
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct", use_fast=True)
sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    max_tokens=1024, # Adjusted from original 1024 to match context
    # n=num_generations, # num_generations is not defined here, assuming 1 or not needed for this script
    logprobs=10, # from original vllm_test.py
    skip_special_tokens=False # from original vllm_test.py
)

def process_output(output):
    logprobs_data = []
    if output.logprobs:
        for token_info in output.logprobs:
            item = {
                "token": token_info.token,
                "logprob": token_info.logprob,
            }
            if token_info.top_logprobs:
                item["top_logprobs"] = [
                    {"token": top.token, "logprob": top.logprob}
                    for top in token_info.top_logprobs
                ]
            logprobs_data.append(item)
    
    return {
        "prompt": output.prompt,
        "response": output.outputs[0].text,
        "logprobs": logprobs_data
    }

def main():
    # prompts = ["Explain transformers in 10-sentences"] * 2000  # demo
    raw_prompts = ["Explain transformers in 10-sentences"] * 2000  # demo

    # Define chat template
    chat_prompts = []
    for raw_prompt in raw_prompts:
        chat = [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {"role": "user", "content": raw_prompt.strip()},
        ]
        chat_prompts.append(tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        ))
    
    # Let vLLM handle batching internally
    outputs = llm.generate(chat_prompts, sampling_params)
    
    # Process outputs with progress bar
    responses = list(tqdm(
        map(process_output, outputs),
        total=len(chat_prompts),
        desc="Processing outputs"
    ))
    
    # Save results to JSON file
    with open('qa_results.json', 'w') as f:
        json.dump(responses, f, indent=2)
    
    return responses

if __name__ == "__main__":
    responses = main()