#!/usr/bin/env python3
"""
simpleqa_analysis.py

Analyze SimpleQA dataset by extracting questions and plotting ECE vs frequency
of text in OLMO training data (pretraining, mid-training, post-training).

Usage:
    python simpleqa_analysis.py --input simpleqa_records.json --output analysis_results.png
"""

import argparse
import json
import re
import time
import random
from typing import Dict, Any, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import requests
import traceback
import os
from openai import OpenAI
from dotenv import load_dotenv

# Add infini-gram package imports
from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer

load_dotenv()

API_URL = "https://api.infini-gram.io/"

# OLMO index mappings for different training stages
OLMO_INDEXES = {
    "pretraining": "v4_olmo-2-1124-13b-instruct_llama",  # Full pretraining data (API)
    "post_training": "index/v4_olmo-2-1124-13b-anneal-adapt_llama",  # Post training data (local package)
}

# Global variables for tokenizer and engine
tokenizer = None
post_training_engine = None

def initialize_local_engine():
    """Initialize the tokenizer and local engine for post_training index."""
    global tokenizer, post_training_engine
    
    if tokenizer is None:
        print("Initializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf", 
            add_bos_token=False, 
            add_eos_token=False
        )
    
    if post_training_engine is None:
        print("Initializing post_training engine...")
        post_training_engine = InfiniGramEngine(
            index_dir=OLMO_INDEXES["post_training"], 
            eos_token_id=tokenizer.eos_token_id
        )

def post_request(payload: Dict[str, Any], retries: int = 3, backoff: float = 0.5) -> Dict:
    """POST request with retry logic."""
    for attempt in range(retries):
        try:
            r = requests.post(API_URL, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(data["error"])
            return data
        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed after {retries} attempts: {e}")
                raise
            time.sleep(backoff * (2 ** attempt))
    return {}

def extract_question_text(full_question: str) -> str:
    """Extract the actual question from the full prompt."""
    # Look for text after "Question: " and before "Your response"
    pattern = r"Question:\s*(.*?)\s*(?:Your response|$)"
    match = re.search(pattern, full_question, re.DOTALL | re.IGNORECASE)
    
    if match:
        question = match.group(1).strip()
        # Remove any trailing instructions or formatting
        question = re.sub(r'\n\s*$', '', question)
        return question
    
    # Fallback: return the full question if pattern doesn't match
    return full_question

def extract_concept(question: str, api_key: str = None) -> str:
    """
    Extract the main concept from a question using GPT-4o.
    
    Args:
        question: The question to analyze
        api_key: OpenAI API key (if None, will use OPENAI_API_KEY env var)
    
    Returns:
        A 2-5 word phrase representing the main concept
    """
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
                    Do not include any other text or explanation. Output only the phrase. Howvever - you must be very specific with the concept.
                    Be very specific to the topic. If there are names in the concept - output in a case-sensitive manner. But if you have common stuff - output case-insensitive

                    Example:
                    Question: What day, month, and year was Kris Cuppens born?

                    Response: Kris Cuppens birth
                    =============================

                    Question: What type of eclipse occurred on August 28, 1802, at 51.3\u00b0N, 105.7\u00b0E?

                    Response: eclipse on August 28
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

def count_ngram(index: str, text: str) -> int:
    """Count occurrences of text in the specified index."""
    
    # Check if this is the post_training index (use local package)
    if index == OLMO_INDEXES["post_training"]:
        try:
            # Initialize engine if needed
            initialize_local_engine()
            print("Local engine initialized")
            
            # Tokenize the text
            input_ids = tokenizer.encode(text)
            print(f"Tokenized '{text}' to: {input_ids}")
            
            # Count using local engine
            result = post_training_engine.count(input_ids=input_ids)
            print(f"Local engine result: {result}")
            
            return int(result.get("count", 0))
            
        except Exception as e:
            print(f"Exception in local count_ngram: {e}")
            traceback.print_exc()
            return 0
    
    # Use API for other indexes (pretraining)
    else:
        payload = {"index": index, "query_type": "count", "query": text}
        print(f"API Payload: {payload}")
        try:
            result = post_request(payload)
            print(f"API Result: {result}")
            
            # Check for error key first as per API documentation
            if "error" in result:
                print(f"API Error: {result['error']}")
                return 0
                
            return int(result.get("count", 0))
        except Exception as e:
            print(f"Exception in API count_ngram: {e}")
            return 0

def get_frequency_counts(question: str) -> Dict[str, int]:
    """Get frequency counts for a question across different training stages."""
    counts = {}
    for stage, index in OLMO_INDEXES.items():
        counts[stage] = count_ngram(index, question)
        time.sleep(0.1)  # Rate limiting
    return counts

def calculate_ece(confidences: List[float], correct_flags: List[bool], n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Determine if sample is in bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = correct_flags[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def analyze_data(input_file: str) -> Tuple[Dict, List]:
    """Analyze the SimpleQA data and return frequency bins and ECE data."""
    print(f"Loading data from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Limit to first 10 records for testing
    data = data[:10]
    print(f"Processing first 10 records (out of {len(data)} total)")
    
    # Extract and process each record
    results = []
    
    for i, record in enumerate(data):
        print(f"Processing record {i+1}/{len(data)}")
        
        # Extract question text
        question = extract_question_text(record["question"])
        print(f"Question: {question}")
        
        # Extract concept using OpenAI
        concept = extract_concept(question)
        print(f"Concept: {concept}")
        
        # Get frequency counts
        freq_counts = get_frequency_counts(concept)
        exit()
        # Store result
        result = {
            "idx": record.get("idx", i),
            "question": question,
            "concept": concept,
            "p_true": record["p_true"],
            "correct": record["correct"],
            "frequency_counts": freq_counts
        }
        results.append(result)
        
        # Add some delay to avoid hitting rate limits
        time.sleep(0.2)
    
    print("Binning data by frequency...")
    
    # Create frequency bins and calculate ECE for each stage
    analysis_results = {}
    
    for stage in OLMO_INDEXES.keys():
        print(f"Analyzing {stage} stage...")
        
        # Get frequency counts for this stage
        frequencies = [r["frequency_counts"][stage] for r in results]
        confidences = np.array([r["p_true"] for r in results])
        correct_flags = np.array([r["correct"] for r in results])
        
        # Create frequency bins (log scale)
        max_freq = max(frequencies) if max(frequencies) > 0 else 1
        freq_bins = np.logspace(0, np.log10(max_freq + 1), 8)  # 7 bins
        
        bin_data = []
        for i in range(len(freq_bins) - 1):
            bin_mask = (np.array(frequencies) >= freq_bins[i]) & (np.array(frequencies) < freq_bins[i + 1])
            
            if bin_mask.sum() > 0:
                bin_confidences = confidences[bin_mask]
                bin_correct = correct_flags[bin_mask]
                
                ece = calculate_ece(bin_confidences, bin_correct)
                avg_freq = np.mean(np.array(frequencies)[bin_mask])
                
                bin_data.append({
                    "avg_frequency": avg_freq,
                    "ece": ece,
                    "count": bin_mask.sum(),
                    "bin_range": (freq_bins[i], freq_bins[i + 1])
                })
        
        analysis_results[stage] = bin_data
    
    return analysis_results, results

def plot_results(analysis_results: Dict, output_file: str):
    """Plot ECE vs frequency for different training stages."""
    plt.figure(figsize=(12, 8))
    
    colors = {"pretraining": "blue", "post_training": "orange"}
    markers = {"pretraining": "o", "post_training": "s"}
    
    for stage, data in analysis_results.items():
        if not data:
            continue
            
        frequencies = [d["avg_frequency"] for d in data]
        eces = [d["ece"] for d in data]
        counts = [d["count"] for d in data]
        
        # Size points by number of samples in bin
        sizes = [min(c * 5, 200) for c in counts]  # Scale point sizes
        
        plt.scatter(frequencies, eces, 
                   c=colors[stage], marker=markers[stage], 
                   s=sizes, alpha=0.7, label=f"{stage.replace('_', ' ').title()}")
        
        # Add trend line
        if len(frequencies) > 1:
            z = np.polyfit(np.log(np.array(frequencies) + 1), eces, 1)
            p = np.poly1d(z)
            x_smooth = np.logspace(np.log10(min(frequencies)), np.log10(max(frequencies)), 100)
            plt.plot(x_smooth, p(np.log(x_smooth + 1)), 
                    color=colors[stage], alpha=0.5, linestyle='--')
    
    plt.xscale('log')
    plt.xlabel('Average Frequency in Training Data (log scale)')
    plt.ylabel('Expected Calibration Error (ECE)')
    plt.title('ECE vs Text Frequency in OLMO Training Data\n(Point size indicates number of samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add text box with summary
    plt.text(0.02, 0.98, 
             f"Analysis of {sum(len(d) for d in analysis_results.values())} frequency bins\n"
             f"Larger points = more samples in bin", 
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze SimpleQA data with infini-gram")
    parser.add_argument("--input", required=True, help="Input SimpleQA JSON file")
    parser.add_argument("--output", default="ece_vs_frequency.png", help="Output plot file")
    parser.add_argument("--save-results", help="Save analysis results to JSON file")
    
    args = parser.parse_args()
    
    # Analyze the data
    analysis_results, raw_results = analyze_data(args.input)
    
    # Save results if requested
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump({
                "analysis": analysis_results,
                "raw_results": raw_results
            }, f, indent=2)
        print(f"Results saved to {args.save_results}")
    
    # Plot results
    plot_results(analysis_results, args.output)
    
    # Print summary
    print("\nSummary:")
    for stage, data in analysis_results.items():
        if data:
            avg_ece = np.mean([d["ece"] for d in data])
            print(f"{stage.replace('_', ' ').title()}: {len(data)} bins, avg ECE = {avg_ece:.4f}")

if __name__ == "__main__":
    main() 