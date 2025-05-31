#!/usr/bin/env python3
"""
simpleqa_analysis.py

Analyze SimpleQA dataset by extracting questions and plotting model confidence (p_true)
vs frequency of concepts in OLMO training data (pretraining and post-training).

Usage:
    python simpleqa_analysis.py --input simpleqa_records.json --output p_true_vs_frequency.png
"""

import argparse
import json
import re
import time
import os
from typing import Dict, Any, List
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import query functions from new modules
from infini_gram_api import query_pretraining_api
from infini_gram_local import query_post_training_local

# Import utils
from utils import load_simpleqa_data, plot_confidence_vs_frequency, save_results, load_and_compare_zs_cot_data

load_dotenv()


def extract_question_text(full_question: str) -> str:
    """Extract the actual question from the full prompt."""
    pattern = r"Question:\s*(.*?)\s*(?:Your response|$)"
    match = re.search(pattern, full_question, re.DOTALL | re.IGNORECASE)
    
    if match:
        question = match.group(1).strip()
        question = re.sub(r'\n\s*$', '', question)
        return question
    
    return full_question


def extract_concept(question: str, api_key: str = None) -> str:
    """Extract the main concept from a question using GPT-4o."""
    try:
        client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": """
                    You are given a question. 
                    Extract the specific concept the question is about. Respond with only a 2-5 word phrase that captures the main concept. 
                    Do not include any other text or explanation. Output only the phrase. However - you must be very specific with the concept.
                    Be very specific to the topic. If there are names in the concept - output in a case-sensitive manner. But if you have common stuff - output case-insensitive.
                    Focus on the key entity, event, or topic that would be most searchable in text.

                    Example:
                    Question: What day, month, and year was Kris Cuppens born?

                    Response: Kris Cuppens birth
                    =============================

                    Question: What type of eclipse occurred on August 28, 1802, at 51.3°N, 105.7°E?

                    Response: eclipse August 28 1802
                    """
                },
                {
                    "role": "user", 
                    "content": f"Question: {question}"
                }
            ],
            max_tokens=100,
            temperature=0.1
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error extracting concept: {e}")
        return "unknown concept"


def generate_semantic_variations(concept: str, api_key: str = None) -> List[str]:
    """Use GPT-4o to generate semantic variations of a concept."""
    try:
        client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": """
                    You are given a concept phrase. Generate 12-15 semantic variations of this concept that would be likely to appear in text corpora, while preserving the core meaning.

                    Allow style variations, abstractions, different phrasings, word order changes, and removal of middle names or unnecessary details. Be flexible and creative.

                    Non-exhaustive list of example transformations:
                    - "2010 David P. Robbins Prize recipient" -> "David Robbins Prize", "Robbins Prize 2010", "David P. Robbins Prize"
                    - "eclipse August 28 1802" -> "1802 eclipse", "August 1802 eclipse", "solar eclipse 1802"
                    - "International Dota 2 2016" -> "Dota 2 International", "2016 International", "Dota 2 2016", "Dota 2016"

                    Include variations with different levels of detail, alternative terms, abbreviations, and different grammatical structures.

                    Return ONLY a JSON list of strings, with no other text.
                    """
                },
                {
                    "role": "user", 
                    "content": concept
                }
            ],
            max_tokens=2048,
            temperature=0.3
        )
        
        content = response.choices[0].message.content.strip()
        variations = json.loads(content)
        
        if concept not in variations:
            variations.insert(0, concept)
            
        return variations[:15]
        
    except Exception as e:
        print(f"Error generating semantic variations: {e}")
        # Fallback to simple variations
        variations = [concept, concept.lower(), concept.title()]
        return list(dict.fromkeys(variations))  # Remove duplicates


def process_record(record, idx, total):
    """Process a single SimpleQA record with concept extraction and frequency queries."""
    print(f"\n{'='*60}")
    print(f"Processing record {idx+1}/{total}")
    print(f"{'='*60}")
    
    # Extract question text
    question = extract_question_text(record["question"])
    print(f"Question: {question}")
    
    # Extract concept using OpenAI
    concept = extract_concept(question)
    print(f"Primary concept: {concept}")
    
    # Generate semantic variations
    semantic_variations = generate_semantic_variations(concept)
    print(f"Semantic variations: {semantic_variations}")
    
    # Query pretraining corpus
    print("\nQuerying pretraining corpus...")
    pretraining_result = query_pretraining_api(concept, semantic_variations)
    
    # Query post-training corpus
    print("\nQuerying post-training corpus...")
    post_training_result = query_post_training_local(concept, semantic_variations)
    
    # Store result
    result = {
        "idx": record.get("idx", idx),
        "question": question,
        "concept": concept,
        "p_true": record["p_true"],
        "correct": record["correct"],
        "pretraining_freq": sum(data.get("doc_count", 0) for data in pretraining_result["all_variations_data"].values()),
        "posttraining_freq": post_training_result["total_doc_count"],
        "combined_freq": sum(data.get("doc_count", 0) for data in pretraining_result["all_variations_data"].values()) + post_training_result["total_doc_count"],
        "pretraining_ngram_count": pretraining_result["total_ngram_count"],
        "posttraining_ngram_count": post_training_result["total_ngram_count"],
        "combined_ngram_count": pretraining_result["total_ngram_count"] + post_training_result["total_ngram_count"],
        "semantic_variations": semantic_variations,
        "pretraining_details": pretraining_result,
        "post_training_details": post_training_result,
        "pretraining_contexts": list(contexts for variation_contexts in pretraining_result["variations_contexts"].values() for contexts in variation_contexts),
        "posttraining_contexts": post_training_result.get("document_contexts", [])
    }
    
    return result


def analyze_data(input_file: str, limit: int = None, overconfident: bool = False, underconfident: bool = False, zs_cot_compare: bool = False, cot_file: str = None) -> List[Dict]:
    """Analyze the SimpleQA data and return results."""
    # Load data using utils with filtering options
    if zs_cot_compare and cot_file:
        # For ZS vs CoT comparison, we need to run both scenarios
        print("Running ZS vs CoT comparison analysis...")
        
        # Scenario 1: CoT correct, ZS incorrect
        print("\n" + "="*60)
        print("SCENARIO 1: CoT Correct, ZS Incorrect")
        print("="*60)
        data_scenario1 = load_and_compare_zs_cot_data(input_file, cot_file, limit, cot_correct_zs_incorrect=True, cot_incorrect_zs_correct=False)
        
        # Scenario 2: CoT incorrect, ZS correct  
        print("\n" + "="*60)
        print("SCENARIO 2: CoT Incorrect, ZS Correct")
        print("="*60)
        data_scenario2 = load_and_compare_zs_cot_data(input_file, cot_file, limit, cot_correct_zs_incorrect=False, cot_incorrect_zs_correct=True)
        
        # Process both scenarios and return combined results with scenario markers
        all_results = []
        
        # Process scenario 1
        if data_scenario1:
            print(f"\nProcessing {len(data_scenario1)} records for Scenario 1...")
            results_scenario1 = []
            with ThreadPoolExecutor(max_workers=20) as executor:
                future_to_idx = {executor.submit(process_record, record, i, len(data_scenario1)): i for i, record in enumerate(data_scenario1)}
                
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        result["comparison_scenario"] = "cot_correct_zs_incorrect"
                        results_scenario1.append(result)
                    except Exception as exc:
                        print(f"Scenario 1 Record {idx} generated an exception: {exc}")
            
            results_scenario1.sort(key=lambda x: x["idx"])
            all_results.extend(results_scenario1)
        
        # Process scenario 2
        if data_scenario2:
            print(f"\nProcessing {len(data_scenario2)} records for Scenario 2...")
            results_scenario2 = []
            with ThreadPoolExecutor(max_workers=20) as executor:
                future_to_idx = {executor.submit(process_record, record, i, len(data_scenario2)): i for i, record in enumerate(data_scenario2)}
                
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        result["comparison_scenario"] = "cot_incorrect_zs_correct"
                        results_scenario2.append(result)
                    except Exception as exc:
                        print(f"Scenario 2 Record {idx} generated an exception: {exc}")
            
            results_scenario2.sort(key=lambda x: x["idx"])
            all_results.extend(results_scenario2)
        
        return all_results
    else:
        # Standard single-file analysis
        data = load_simpleqa_data(input_file, limit, overconfident, underconfident)
    
        results = []
        
        # Process records in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=20) as executor:
            # Submit all tasks
            future_to_idx = {executor.submit(process_record, record, i, len(data)): i for i, record in enumerate(data)}
            
            # Process results as they complete
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    print(f"Record {idx} generated an exception: {exc}")
        
        # Sort results by original index
        results.sort(key=lambda x: x["idx"])
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY OF ALL RESULTS:")
        print("="*60)
        
        for i, result in enumerate(results):
            print(f"Record {i+1}: {result['concept']}")
            print(f"  p_true: {result['p_true']:.3f}")
            print(f"  pretraining_freq: {result['pretraining_freq']}")
            print(f"  posttraining_freq: {result['posttraining_freq']}")
            print(f"  combined_freq: {result['combined_freq']}")
            print(f"  pretraining_ngram_count: {result['pretraining_ngram_count']}")
            print(f"  posttraining_ngram_count: {result['posttraining_ngram_count']}")
            print(f"  combined_ngram_count: {result['combined_ngram_count']}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Analyze SimpleQA data: plot p_true vs document frequency in training data")
    parser.add_argument("--input", required=True, help="Input SimpleQA JSON file (zero-shot file for ZS vs CoT comparison)")
    parser.add_argument("--output", default="p_true_vs_frequency.png", help="Output plot file")
    parser.add_argument("--save-results", help="Save analysis results to JSON file")
    parser.add_argument("--limit", type=int, help="Limit number of records to process")
    parser.add_argument("--overconfident", action="store_true", help="Filter for overconfident cases: p_true > 0.7 and correct = False")
    parser.add_argument("--underconfident", action="store_true", help="Filter for underconfident cases: p_true < 0.3 and correct = True")
    parser.add_argument("--zs-cot-compare", action="store_true", help="Run ZS vs CoT comparison analysis")
    parser.add_argument("--cot-file", help="Input CoT JSON file for ZS vs CoT comparison")
    
    args = parser.parse_args()
    
    # Check for mutually exclusive filtering options
    filter_options = [args.overconfident, args.underconfident, args.zs_cot_compare]
    if sum(filter_options) > 1:
        print("Error: --overconfident, --underconfident, and --zs-cot-compare are mutually exclusive")
        return
    
    # Check if zs_cot_compare requires cot_file
    if args.zs_cot_compare and not args.cot_file:
        print("Error: --zs-cot-compare requires --cot-file to be specified")
        return
    
    # Analyze the data
    results = analyze_data(args.input, args.limit, args.overconfident, args.underconfident, args.zs_cot_compare, args.cot_file)
    
    # Save results if requested
    if args.save_results:
        save_results({"raw_results": results}, args.save_results)
    
    # Plot results using utils
    if args.zs_cot_compare:
        # Separate results by scenario and create plots for each
        scenario1_results = [r for r in results if r.get("comparison_scenario") == "cot_correct_zs_incorrect"]
        scenario2_results = [r for r in results if r.get("comparison_scenario") == "cot_incorrect_zs_correct"]
        
        print(f"\nCreating plots for {len(scenario1_results)} CoT Correct/ZS Incorrect cases...")
        if scenario1_results:
            plot_confidence_vs_frequency(scenario1_results, args.output, 
                                        cot_correct_zs_incorrect=True)
        
        print(f"\nCreating plots for {len(scenario2_results)} CoT Incorrect/ZS Correct cases...")
        if scenario2_results:
            plot_confidence_vs_frequency(scenario2_results, args.output, 
                                        cot_incorrect_zs_correct=True)
    else:
        plot_confidence_vs_frequency(results, args.output, args.overconfident, args.underconfident)
    
    # Print final summary
    if results:
        print(f"\nProcessed {len(results)} questions")
        
        if args.zs_cot_compare:
            # Print summary for each scenario
            scenario1_results = [r for r in results if r.get("comparison_scenario") == "cot_correct_zs_incorrect"]
            scenario2_results = [r for r in results if r.get("comparison_scenario") == "cot_incorrect_zs_correct"]
            
            if scenario1_results:
                print(f"\nScenario 1 (CoT Correct, ZS Incorrect): {len(scenario1_results)} cases")
                avg_p_true_s1 = np.mean([r["p_true"] for r in scenario1_results])
                avg_combined_freq_s1 = np.mean([r["combined_freq"] for r in scenario1_results])
                avg_combined_ngram_s1 = np.mean([r["combined_ngram_count"] for r in scenario1_results])
                print(f"  Average p_true: {avg_p_true_s1:.4f}")
                print(f"  Average combined frequency: {avg_combined_freq_s1:.1f}")
                print(f"  Average combined n-gram count: {avg_combined_ngram_s1:.1f}")
            
            if scenario2_results:
                print(f"\nScenario 2 (CoT Incorrect, ZS Correct): {len(scenario2_results)} cases")
                avg_p_true_s2 = np.mean([r["p_true"] for r in scenario2_results])
                avg_combined_freq_s2 = np.mean([r["combined_freq"] for r in scenario2_results])
                avg_combined_ngram_s2 = np.mean([r["combined_ngram_count"] for r in scenario2_results])
                print(f"  Average p_true: {avg_p_true_s2:.4f}")
                print(f"  Average combined frequency: {avg_combined_freq_s2:.1f}")
                print(f"  Average combined n-gram count: {avg_combined_ngram_s2:.1f}")
        else:
            # Standard summary
            avg_p_true = np.mean([r["p_true"] for r in results])
            avg_pre_freq = np.mean([r["pretraining_freq"] for r in results])
            avg_post_freq = np.mean([r["posttraining_freq"] for r in results])
            avg_combined_freq = np.mean([r["combined_freq"] for r in results])
            avg_pre_ngram = np.mean([r["pretraining_ngram_count"] for r in results])
            avg_post_ngram = np.mean([r["posttraining_ngram_count"] for r in results])
            avg_combined_ngram = np.mean([r["combined_ngram_count"] for r in results])
            print(f"Average p_true: {avg_p_true:.4f}")
            print(f"Average pretraining frequency: {avg_pre_freq:.1f}")
            print(f"Average post-training frequency: {avg_post_freq:.1f}")
            print(f"Average combined frequency: {avg_combined_freq:.1f}")
            print(f"Average pretraining n-gram count: {avg_pre_ngram:.1f}")
            print(f"Average post-training n-gram count: {avg_post_ngram:.1f}")
            print(f"Average combined n-gram count: {avg_combined_ngram:.1f}")


if __name__ == "__main__":
    main() 