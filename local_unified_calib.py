import numpy as np
from datasets import load_dataset
import re
import pandas as pd
from tqdm.auto import tqdm
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
import traceback
from openai import OpenAI
from dotenv import load_dotenv
from prompt_templates import (
    format_gsm8k_question,
    format_mmlu_question,
    format_medmcqa_question,
    format_simpleqa_question
)
from answer_parsers import (
    parse_gsm8k_answer,
    parse_mmlu_answer,
    parse_medmcqa_answer,
    parse_simpleqa_answer
)
from answer_validators import (
    validate_gsm8k_answer,
    validate_mmlu_answer, 
    validate_medmcqa_answer,
    validate_simpleqa_answer
)

load_dotenv()

# OpenAI configuration for SimpleQA grading
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Server configuration
url = "http://localhost:8000/v1/chat/completions"
headers = {"Content-Type": "application/json"}

TRUE_SYNS = [
    "true",
    "correct",
    "truth",
    "yes",
    "right",
    "verdade",
]

FALSE_SYNS = [
    "false",
    "incorrect",
    "wrong",
    "fake",
    "no",
    "not",
    "none",
]

def send_request(payload):
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Request error: {e}")
        return None

def prepare_questions(dataset, dataset_split, dataset_type, indices=None):
    """Process the dataset and prepare all questions"""
    questions = []
    
    if indices is not None:
        idx_range = indices
    else:
        idx_range = range(len(dataset[dataset_split]))
    
    for idx in tqdm(idx_range, desc="Preparing questions"):
        try:
            if dataset_type == "simpleqa":
                question = dataset[dataset_split][idx]["problem"]
            else:
                question = dataset[dataset_split][idx]["question"]
            
            if dataset_type == "gsm8k":
                question_str = format_gsm8k_question(question)
                true_answer = int(dataset[dataset_split][idx]["answer"].split("\n")[-1][5:].replace(",", ""))
            elif dataset_type == "mmlu":
                choices_list = dataset[dataset_split][idx]["choices"]
                subject = dataset[dataset_split][idx]["subject"].replace("_", " ")
                question_str = format_mmlu_question(question, choices_list, subject)
                true_answer_idx = dataset[dataset_split][idx]["answer"]
                true_answer = ["A", "B", "C", "D"][true_answer_idx]
            elif dataset_type == "medmcqa":
                choices_list = [
                    dataset[dataset_split][idx]["opa"],
                    dataset[dataset_split][idx]["opb"],
                    dataset[dataset_split][idx]["opc"],
                    dataset[dataset_split][idx]["opd"]
                ]
                question_str = format_medmcqa_question(question, choices_list)
                true_answer = chr(65 + dataset[dataset_split][idx]["cop"])
            elif dataset_type == "simpleqa":
                question_str = format_simpleqa_question(question)
                true_answer = dataset[dataset_split][idx]["answer"]
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")
            
            questions.append({
                "idx": idx,
                "original_question": question,
                "formatted_question": question_str,
                "true_answer": true_answer,
                "dataset_type": dataset_type
            })
        except Exception as e:
            print(f"Error preparing question {idx}: {e}")
    
    return questions

def generate_responses(questions, exp_type, max_workers):
    """Generate LLM responses for all questions using ThreadPool"""
    
    def generate_single_response(question_data):
        try:
            idx = question_data["idx"]
            question_str = question_data["formatted_question"]
            dataset_type = question_data["dataset_type"]
            
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
            
            # Add chain-of-thought injection for cot_exp
            if exp_type == "cot_exp":
                inject_cot_payload = {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": question_str},
                        {"role": "assistant", "content": response_text},
                        {
                            "role": "user",
                            "content": "Before answering whether your above answer is correct, please provide a detailed chain-of-thought explanation of your reasoning. Explain step-by-step how you arrived at your answer and why you think it is correct or might be incorrect.",
                        },
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1024,
                }

                inject_cot_response = send_request(inject_cot_payload)
                if not inject_cot_response:
                    return None
                inject_cot_text = inject_cot_response["choices"][0]["message"]["content"]
            else:
                inject_cot_text = None

            # Final request for confidence check with logprobs
            final_payload = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question_str},
                    {"role": "assistant", "content": response_text},
                ],
                "temperature": 0.7,
                "max_tokens": 1,
                "n_probs": 25,
            }

            # Add CoT messages if in cot_exp mode
            if exp_type == "cot_exp":
                final_payload["messages"].extend([
                    {
                        "role": "user",
                        "content": "Before answering whether your above answer is correct, please provide a detailed chain-of-thought explanation of your reasoning. Explain step-by-step how you arrived at your answer and why you think it is correct or might be incorrect.",
                    },
                    {"role": "assistant", "content": inject_cot_text},
                ])

            final_payload["messages"].append({
                "role": "user",
                "content": "Is the above answer correct? Answer only with the single word 'true' or 'false'.",
            })

            final_response = send_request(final_payload)
            if not final_response:
                return None

            # Extract logprobs for 'true' and 'false' tokens
            logprobs = final_response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
            probs = {"true": 0.0, "false": 0.0}
            
            for item in logprobs:
                token = item["token"].lower()
                if any(key in token for key in TRUE_SYNS + FALSE_SYNS):
                    if any(key in token for key in TRUE_SYNS):
                        actual_key = "true"
                    elif any(key in token for key in FALSE_SYNS):
                        actual_key = "false"
                    probs[actual_key] += np.exp(item["logprob"])

            p_true = probs["true"] / (probs["true"] + probs["false"])
            
            result = {
                "idx": idx,
                "question": question_str,
                "response": response_text,
                "p_true": p_true,
                "true_answer": question_data["true_answer"],
                "dataset_type": dataset_type,
                "original_question": question_data["original_question"]
            }
            
            if exp_type == "cot_exp":
                result["inject_cot"] = inject_cot_text
                
            return result
            
        except Exception as e:
            print(f"Error generating response for question {question_data['idx']}: {e}")
            traceback.print_exc()
            return None
    
    responses = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_single_response, question) for question in questions]
        
        for future in tqdm(as_completed(futures), total=len(questions), desc="Generating responses"):
            result = future.result()
            if result:
                responses.append(result)
    
    return responses

def parse_and_validate(responses, exp_type, max_workers):
    """Parse and validate all responses using ThreadPool"""
    
    def process_single_response(response_data):
        try:
            if not response_data:
                return None
                
            dataset_type = response_data["dataset_type"]
            response_text = response_data["response"]
            
            # Parse answer based on dataset type
            if dataset_type == "gsm8k":
                answer = parse_gsm8k_answer(response_text)
                is_correct = validate_gsm8k_answer(answer, response_data["true_answer"])
            elif dataset_type == "mmlu":
                answer = parse_mmlu_answer(response_text)
                is_correct = validate_mmlu_answer(answer, response_data["true_answer"])
            elif dataset_type == "medmcqa":
                answer = parse_medmcqa_answer(response_text)
                is_correct = validate_medmcqa_answer(answer, response_data["true_answer"])
            elif dataset_type == "simpleqa":
                answer = parse_simpleqa_answer(response_text)
                is_correct = validate_simpleqa_answer(
                    response_data["original_question"], 
                    answer, 
                    response_data["true_answer"]
                )
            
            result = {
                "idx": response_data["idx"],
                "question": response_data["question"],
                "response": response_data["response"],
                "answer": answer,
                "p_true": response_data["p_true"],
                "true_answer": response_data["true_answer"],
                "correct": is_correct,
            }
            
            if exp_type == "cot_exp" and "inject_cot" in response_data:
                result["inject_cot"] = response_data["inject_cot"]
                
            return result
            
        except Exception as e:
            print(f"Error processing response {response_data.get('idx', 'unknown')}: {e}")
            traceback.print_exc()
            return None
    
    processed_responses = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_response, response) for response in responses]
        
        for future in tqdm(as_completed(futures), total=len(responses), desc="Processing responses"):
            result = future.result()
            if result:
                processed_responses.append(result)
    
    return processed_responses

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model_name",
        choices=["llama", "qwen", "gemma"],
        help="Name of local model to use",
    )
    parser.add_argument(
        "dataset_name",
        choices=["gsm8k", "mmlu", "medmcqa", "simpleqa"],
        help="Dataset to use",
    )
    parser.add_argument(
        "dataset_split",
        choices=["train", "test", "validation"],
        help="Dataset split to use",
    )
    parser.add_argument(
        "exp_type",
        choices=["cot_exp", "zs_exp"],
        help="Experiment type",
    )
    parser.add_argument(
        "max_workers",
        type=int,
        default=12,
        help="Maximum number of workers to use",
    )
    args = parser.parse_args()

    # Load appropriate dataset
    assert not (args.dataset_name == "medmcqa" and args.dataset_split == "test"), "MedMCQA test split does not contain answers - cannot calibrate now, use validation"

    
    if args.dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main")
    elif args.dataset_name == "mmlu":
        dataset = load_dataset("cais/mmlu", "all")
    elif args.dataset_name == "medmcqa":
        dataset = load_dataset("openlifescienceai/medmcqa", "default")
    elif args.dataset_name == "simpleqa":
        dataset = load_dataset("basicv8vc/SimpleQA")

    if args.dataset_split not in dataset:
        raise ValueError(f"Dataset split {args.dataset_split} not found in dataset {args.dataset_name}")
 
    dataset_split = args.dataset_split
    print(f"Dataset: {args.dataset_name}, Split: {dataset_split}")
    
    BASE_DIR = f"newww/{args.exp_type}/{args.dataset_name}"
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"Base directory: {BASE_DIR}")

    indices = None
    if (args.dataset_name == "medmcqa" and args.dataset_split == "train") or (args.dataset_name == "mmlu" and args.dataset_split == "test"):
        np.random.seed(42)
        end_idx = min(5000, len(dataset[dataset_split]))
        indices = [int(i) for i in np.random.choice(len(dataset[dataset_split]), end_idx, replace=False)]

    FINAL_FILE_NAME = f"{BASE_DIR}/{args.exp_type}_records_{args.dataset_split}_full_{args.model_name}.csv"
    assert not os.path.exists(FINAL_FILE_NAME), "Final file already exists - You should delete it before running the experiment"

    # Step 1: Process dataset and prepare questions
    print("Step 1: Processing dataset and preparing questions...")
    all_questions = prepare_questions(dataset, dataset_split, args.dataset_name, indices)
    print(f"Prepared {len(all_questions)} questions")

    # Step 2: Generate LLM responses
    print("Step 2: Generating LLM responses...")
    all_responses = generate_responses(all_questions, args.exp_type, args.max_workers)
    print(f"Generated {len(all_responses)} responses")

    # Step 3: Parse and validate responses
    print("Step 3: Parsing and validating responses...")
    processed_results = parse_and_validate(all_responses, args.exp_type, args.max_workers)
    print(f"Processed {len(processed_results)} results")

    # Save results
    records = processed_results
    if len(records) > 0:
        final_df = pd.DataFrame.from_records(records)
        final_df.to_csv(FINAL_FILE_NAME)
        print(f"Results saved to {FINAL_FILE_NAME}") 