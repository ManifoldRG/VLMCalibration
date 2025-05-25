import json
import os
import re
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm.auto import tqdm

# vLLM
from vllm import LLM, SamplingParams


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)


# ---------------------------------------------------------------------------
# Prompt formatters – identical to local_unified_calib
# ---------------------------------------------------------------------------


def format_gsm8k_question(question: str) -> str:
    return (
        "Given the following problem, reason and give a final answer to the problem.\n"
        f"Problem: {question}\n"
        'Your response should end with "The final answer is [answer]" where [answer] is the response to the problem.'
    )


def format_mmlu_question(question: str, choices_list: list[str], subject: str) -> str:
    formatted_choices = "\n".join(
        [f"{chr(65 + i)}: {choice}" for i, choice in enumerate(choices_list)]
    )
    return (
        f"The following is a multiple choice question (with choices) about {subject}.\n"
        "Reason and give a final answer to the question. Your response should end with 'The final answer is [answer]' where [answer] is solely the letter corresponding to the correct choice.\n"
        f"Question: {question}\n"
        f"Choices:\n{formatted_choices}"
    )


def format_medmcqa_question(question: str, choices_list: list[str]) -> str:
    formatted_choices = "\n".join(
        [f"{chr(65 + i)}: {choice}" for i, choice in enumerate(choices_list)]
    )
    return (
        "Given the following medical question and options, reason and select the correct answer letter (A-D).\n"
        f"Question: {question}\n"
        f"Options:\n{formatted_choices}\n"
        "Your response should end with 'The final answer is [letter]' where [letter] is A, B, C or D."
    )


def format_simpleqa_question(question: str) -> str:
    return (
        "Given the following question, provide a clear and concise answer.\n"
        f"Question: {question}\n\n"
        'Your response should end with "The answer is: [answer]" where [answer] is your complete answer.'
    )


# ---------------------------------------------------------------------------
# Answer parsers – identical to local_unified_calib
# ---------------------------------------------------------------------------


def parse_gsm8k_answer(output: str) -> Optional[int]:
    final_answer_match = re.search(r"The final answer is \$?(\d+,?\d*)", output)
    if final_answer_match:
        return int(final_answer_match.group(1).replace(",", ""))
    matches = re.findall(r"\$?(\d+,?\d*)", output)
    return int(matches[-1].replace(",", "")) if matches else None


def parse_mmlu_answer(output: str) -> Optional[str]:
    match = re.search(r"The final answer is ([A-D])", output)
    return match.group(1) if match else None


def parse_medmcqa_answer(output: str) -> Optional[str]:
    match = re.search(r"The final answer is ([A-D])", output)
    return match.group(1) if match else None


def parse_simpleqa_answer(output: str) -> str:
    answer_match = re.search(
        r"The answer is: (.*?)(?:\.|$)", output, re.IGNORECASE | re.MULTILINE
    )
    return answer_match.group(1).strip() if answer_match else output.strip()


# ---------------------------------------------------------------------------
# Answer validators – identical to local_unified_calib
# ---------------------------------------------------------------------------


def validate_gsm8k_answer(answer: Optional[int], true_answer: int) -> Optional[bool]:
    if answer is None:
        return None
    return abs(answer - true_answer) < 1e-4


def validate_mmlu_answer(answer: Optional[str], true_answer: str) -> Optional[bool]:
    return None if answer is None else answer == true_answer


def validate_medmcqa_answer(answer: Optional[str], true_answer: str) -> Optional[bool]:
    return None if answer is None else answer == true_answer


def validate_simpleqa_answer(question: str, model_answer: str, true_answer: str) -> Optional[bool]:
    """Grade SimpleQA answers with GPT-4-o."""

    if openai_client is None:
        return None

    from simpleQA_grader_template import GRADER_TEMPLATE

    grading_prompt = GRADER_TEMPLATE.format(
        question=question, target=true_answer, predicted_answer=model_answer
    )

    try:
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful grading assistant."},
                {"role": "user", "content": grading_prompt},
            ],
            temperature=0.7,
            max_tokens=1,
        )
        letter = resp.choices[0].message.content.strip()
        if letter == "A":
            return True
        if letter == "B":
            return False
        if letter == "C":
            return "NOT_ATTEMPTED"  # type: ignore[return-value]
    except Exception as exc:
        print(f"Grading error: {exc}")

    return None


# ---------------------------------------------------------------------------
# p_true extraction helper (same logic as local script)
# ---------------------------------------------------------------------------


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


def extract_p_true(top_logprobs: Dict[str, float]) -> float:
    probs = {"true": 0.0, "false": 0.0}
    for token, logprob in top_logprobs.items():
        tok = token.lower().lstrip("Ġ▁ ")
        if any(s in tok for s in TRUE_SYNS):
            probs["true"] += np.exp(logprob)
        elif any(s in tok for s in FALSE_SYNS):
            probs["false"] += np.exp(logprob)

    denom = probs["true"] + probs["false"]
    return 0.5 if denom == 0 else probs["true"] / denom


# ---------------------------------------------------------------------------
# Core processing functions
# ---------------------------------------------------------------------------

def prepare_first_prompts(dataset_type: str, dataset, split: str, indices: List[int]) -> List[Tuple[int, str, Any]]:
    """Prepare the first round of prompts for all questions."""
    first_prompts = []
    
    for idx in indices:
        try:
            # Build question prompt
            if dataset_type == "simpleqa":
                question = dataset[split][idx]["problem"]
            else:
                question = dataset[split][idx]["question"]

            if dataset_type == "gsm8k":
                question_str = format_gsm8k_question(question)
            elif dataset_type == "mmlu":
                choices_list = dataset[split][idx]["choices"]
                subject = dataset[split][idx]["subject"].replace("_", " ")
                question_str = format_mmlu_question(question, choices_list, subject)
            elif dataset_type == "medmcqa":
                choices_list = [
                    dataset[split][idx]["opa"],
                    dataset[split][idx]["opb"],
                    dataset[split][idx]["opc"],
                    dataset[split][idx]["opd"],
                ]
                question_str = format_medmcqa_question(question, choices_list)
            elif dataset_type == "simpleqa":
                question_str = format_simpleqa_question(question)
            else:
                raise ValueError(f"Unknown dataset: {dataset_type}")
                
            # Store original dataset entry for later validation
            metadata = dataset[split][idx]
            
            first_prompts.append((idx, question_str, metadata))
        except Exception as exc:
            print(f"Error preparing question {idx}: {exc}")
            traceback.print_exc()
            
    return first_prompts


def prepare_cot_prompts(first_results: List[Tuple[int, str, str]], question_prompts: List[Tuple[int, str, Any]]) -> List[Tuple[int, str]]:
    """Prepare chain-of-thought prompts from first generation results."""
    cot_prompts = []
    
    for (idx, response_text, _), (_, question_str, _) in zip(first_results, question_prompts):
        inject_cot_payload = (
            "You are a helpful assistant.\n"
            f"{question_str}\n"
            f"{response_text}\n"
            "Before answering whether your above answer is correct, please provide a detailed chain-of-thought explanation of your reasoning. Explain step-by-step how you arrived at your answer and why you think it is correct or might be incorrect."
        )
        cot_prompts.append((idx, inject_cot_payload))
        
    return cot_prompts


def prepare_confidence_prompts(
    first_results: List[Tuple[int, str, str]], 
    question_prompts: List[Tuple[int, str, Any]],
    cot_results: Optional[List[Tuple[int, str, str]]] = None
) -> List[Tuple[int, str]]:
    """Prepare confidence probe prompts."""
    confidence_prompts = []
    
    for i, ((idx, response_text, _), (_, question_str, _)) in enumerate(zip(first_results, question_prompts)):
        messages = [
            "You are a helpful assistant.",
            question_str,
            response_text,
        ]
        
        if cot_results:
            cot_idx, _, cot_text = cot_results[i]
            assert cot_idx == idx, f"Mismatched indices: {cot_idx} vs {idx}"
            messages.extend([
                "Before answering whether your above answer is correct, please provide a detailed chain-of-thought explanation of your reasoning. Explain step-by-step how you arrived at your answer and why you think it is correct or might be incorrect.",
                cot_text,
            ])
            
        messages.append(
            "Is the above answer correct? Answer only with the single word 'true' or 'false'."
        )
        
        confidence_prompts.append((idx, "\n".join(messages)))
        
    return confidence_prompts


def parse_and_validate_results(
    dataset_type: str,
    first_results: List[Tuple[int, str, str]],
    confidence_results: List[Tuple[int, str, Dict[str, float]]],
    question_prompts: List[Tuple[int, str, Any]],
    cot_results: Optional[List[Tuple[int, str, str]]] = None
) -> List[Dict[str, Any]]:
    """Parse answers and validate against ground truth."""
    records = []
    
    for i, ((idx, response_text, _), (conf_idx, _, logprobs), (_, _, metadata)) in enumerate(
        zip(first_results, confidence_results, question_prompts)
    ):
        assert idx == conf_idx, f"Mismatched indices: {idx} vs {conf_idx}"
        
        # Parse answer
        if dataset_type == "gsm8k":
            answer = parse_gsm8k_answer(response_text)
        elif dataset_type == "mmlu":
            answer = parse_mmlu_answer(response_text)
        elif dataset_type == "medmcqa":
            answer = parse_medmcqa_answer(response_text)
        elif dataset_type == "simpleqa":
            answer = parse_simpleqa_answer(response_text)
            
        # Extract confidence
        try:
            p_true = extract_p_true(logprobs)
        except Exception:
            p_true = 0.5
            
        # Validate against ground truth
        if dataset_type == "gsm8k":
            true_answer = int(
                metadata["answer"].split("\n")[-1][5:].replace(",", "")
            )
            is_correct = validate_gsm8k_answer(answer, true_answer)
        elif dataset_type == "mmlu":
            true_answer_idx = metadata["answer"]
            true_answer = ["A", "B", "C", "D"][true_answer_idx]
            is_correct = validate_mmlu_answer(answer, true_answer)
        elif dataset_type == "medmcqa":
            true_answer = chr(65 + metadata["cop"])
            is_correct = validate_medmcqa_answer(answer, true_answer)
        elif dataset_type == "simpleqa":
            true_answer = metadata["answer"]
            # For SimpleQA, we need the question text for validation
            question = metadata["problem"]
            is_correct = validate_simpleqa_answer(question, answer, true_answer)
        else:
            true_answer = None
            is_correct = None
            
        # Build result record
        result = {
            "question": question_prompts[i][1],  # question_str
            "response": response_text,
            "answer": answer,
            "p_true": p_true,
            "true_answer": true_answer,
            "correct": is_correct,
        }
        
        if cot_results:
            cot_idx, _, cot_text = cot_results[i]
            assert cot_idx == idx, f"Mismatched indices: {cot_idx} vs {idx}"
            result["inject_cot"] = cot_text
            
        records.append(result)
        
    return records


# ---------------------------------------------------------------------------
# vLLM generation helper
# ---------------------------------------------------------------------------

def batch_vllm_generate(
    llm: LLM,
    prompts: List[Tuple[int, str]],
    temperature: float = 0.7,
    max_tokens: int = 1024,
    logprobs: Optional[int] = None,
    top_p: float = 0.95,
) -> List[Tuple[int, str, Any]]:
    """Generate text for a batch of prompts using vLLM."""
    prompt_texts = [prompt for _, prompt in prompts]
    prompt_ids = [idx for idx, _ in prompts]
    
    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        logprobs=logprobs,
    )
    
    outputs = llm.generate(prompt_texts, params)
    
    # Match outputs with their indices
    results = []
    for idx, output in zip(prompt_ids, outputs):
        if logprobs is not None and output.outputs[0].logprobs:
            # Extract logprobs from the first token
            token_logprobs = output.outputs[0].top_logprobs[0]
            results.append((idx, output.outputs[0].text, token_logprobs))
        else:
            results.append((idx, output.outputs[0].text, None))
            
    return results


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="HF model name/path for vLLM")
    parser.add_argument("dataset_name", choices=["gsm8k", "mmlu", "medmcqa", "simpleqa"], type=str)
    parser.add_argument("dataset_split", choices=["train", "validation", "test"], type=str)
    parser.add_argument("exp_type", choices=["cot_exp", "zs_exp"], type=str)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for vLLM generation")
    args = parser.parse_args()

    # -------------------------- dataset loading ---------------------------
    if args.dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main")
    elif args.dataset_name == "mmlu":
        dataset = load_dataset("cais/mmlu", "all")
    elif args.dataset_name == "medmcqa":
        dataset = load_dataset("openlifescienceai/medmcqa", "default")
    elif args.dataset_name == "simpleqa":
        dataset = load_dataset("basicv8vc/SimpleQA")

    if args.dataset_split not in dataset:
        raise ValueError(f"Split {args.dataset_split} not found in dataset {args.dataset_name}")

    if args.dataset_name == "medmcqa" and args.dataset_split == "test":
        raise ValueError("MedMCQA test split lacks answers – cannot calibrate.")

    # ------------------------------ model ---------------------------------
    llm = LLM(
        model=args.model_name,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )

    # -------------------------- index selection ---------------------------
    split = args.dataset_split
    total_examples = len(dataset[split])

    if (
        args.dataset_name == "medmcqa" and split == "train"
    ) or (
        args.dataset_name == "mmlu" and split == "test"
    ):
        np.random.seed(42)
        total_examples = min(5000, total_examples)
        indices = list(np.random.choice(len(dataset[split]), total_examples, replace=False))
    else:
        indices = list(range(total_examples))

    # ----------------------------- I/O paths -----------------------------
    base_dir = f"{args.exp_type}/{args.dataset_name}"
    os.makedirs(base_dir, exist_ok=True)

    model_stub = os.path.basename(args.model_name).replace("/", "_")
    final_file = f"{base_dir}/{args.exp_type}_records_{split}_full_{model_stub}.csv"
    partial_file = f"{base_dir}/{args.exp_type}_records_{split}_partial_{model_stub}.csv"

    if os.path.exists(final_file):
        raise FileExistsError("Final output CSV already exists - delete it before rerunning.")

    # -------------------------- Batch Processing --------------------------
    print("Preparing question prompts...")
    question_prompts = prepare_first_prompts(args.dataset_name, dataset, split, indices)
    
    # Process in batches to avoid memory issues
    all_records = []
    for batch_start in tqdm(range(0, len(question_prompts), args.batch_size), desc="Processing batches"):
        batch_end = min(batch_start + args.batch_size, len(question_prompts))
        batch_prompts = question_prompts[batch_start:batch_end]
        
        # First generation pass
        print(f"Batch {batch_start//args.batch_size + 1}: Running first generation pass...")
        first_batch = [(idx, prompt) for idx, prompt, _ in batch_prompts]
        first_results = batch_vllm_generate(llm, first_batch)
        
        # Optional CoT generation
        cot_results = None
        if args.exp_type == "cot_exp":
            print(f"Batch {batch_start//args.batch_size + 1}: Running chain-of-thought generation...")
            cot_prompts = prepare_cot_prompts(first_results, batch_prompts)
            cot_results = batch_vllm_generate(llm, cot_prompts)
        
        # Confidence probe generation
        print(f"Batch {batch_start//args.batch_size + 1}: Running confidence probe generation...")
        confidence_prompts = prepare_confidence_prompts(first_results, batch_prompts, cot_results)
        confidence_results = batch_vllm_generate(
            llm, confidence_prompts, max_tokens=1, logprobs=25, top_p=1
        )
        
        # Parse and validate
        print(f"Batch {batch_start//args.batch_size + 1}: Parsing and validating results...")
        batch_records = parse_and_validate_results(
            args.dataset_name, first_results, confidence_results, batch_prompts, cot_results
        )
        
        all_records.extend(batch_records)
        
        # Save partial results
        if all_records:
            pd.DataFrame.from_records(all_records).to_csv(partial_file, index=False)
    
    # Save final results
    if all_records:
        pd.DataFrame.from_records(all_records).to_csv(final_file, index=False)
        if os.path.exists(partial_file):
            os.remove(partial_file)
    
    print(f"Completed processing {len(all_records)} records.")


if __name__ == "__main__":
    main()
