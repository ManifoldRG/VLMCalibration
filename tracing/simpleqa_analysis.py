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

# Import query functions from new modules
from infini_gram_api import query_pretraining_api
from infini_gram_local import query_post_training_local

# Import utils
from utils import load_simpleqa_data, plot_confidence_vs_frequency, save_results

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
                    You are given a concept phrase. Generate 8-10 semantic variations of this concept that would be likely to appear in text corpora, while preserving the core meaning and specificity.

                    CRITICAL: Keep the concept specific and meaningful. Do NOT make it overly general or broad.
                    CRITICAL: Do NOT generate individual words or very partial phrases from the concept.
                    CRITICAL: Each variation should refer to the same core concept, but can have slightly different levels of detail.

                    Think about:
                    - Different phrasings of the same concept
                    - Different word orders of the same information
                    - Alternative ways to express the same details
                    - Preserve key identifying information (names, dates, locations, etc.) but allow some flexibility
                    - Different grammatical structures for the same concept
                    - How the same concept might appear in different text styles
                    - Slightly shorter or longer versions that maintain the core meaning

                    Return ONLY a JSON list of strings, with no other text. Each variation should maintain the core concept meaning.

                    Example:
                    Input: "eclipse August 28 1802"
                    Output: ["eclipse August 28 1802", "August 28 1802 eclipse", "eclipse on August 28 1802", "eclipse August 28th 1802", "1802 August 28 eclipse", "eclipse of August 28 1802", "28 August 1802 eclipse", "eclipse on August 28", "August 28 eclipse", "solar eclipse August 28 1802"]
                    """
                },
                {
                    "role": "user", 
                    "content": concept
                }
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        content = response.choices[0].message.content.strip()
        variations = json.loads(content)
        
        if concept not in variations:
            variations.insert(0, concept)
            
        return variations[:10]
        
    except Exception as e:
        print(f"Error generating semantic variations: {e}")
        # Fallback to simple variations
        variations = [concept, concept.lower(), concept.title()]
        return list(dict.fromkeys(variations))  # Remove duplicates


def analyze_data(input_file: str, limit: int = None) -> List[Dict]:
    """Analyze the SimpleQA data and return results."""
    # Load data using utils
    data = load_simpleqa_data(input_file, limit)
    
    results = []
    
    for i, record in enumerate(data):
        print(f"\n{'='*60}")
        print(f"Processing record {i+1}/{len(data)}")
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
            "idx": record.get("idx", i),
            "question": question,
            "concept": concept,
            "p_true": record["p_true"],
            "correct": record["correct"],
            "pretraining_freq": sum(data.get("doc_count", 0) for data in pretraining_result["all_variations_data"].values()),
            "posttraining_freq": post_training_result["doc_count"],
            "combined_freq": sum(data.get("doc_count", 0) for data in pretraining_result["all_variations_data"].values()) + post_training_result["doc_count"],
            "semantic_variations": semantic_variations,
            "pretraining_details": pretraining_result,
            "post_training_details": post_training_result,
            "pretraining_contexts": list(contexts for variation_contexts in pretraining_result["variations_contexts"].values() for contexts in variation_contexts),
            "posttraining_contexts": post_training_result.get("document_contexts", [])
        }
        results.append(result)
        
        # Add delay to avoid rate limits
        time.sleep(0.3)
    
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
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze SimpleQA data: plot p_true vs document frequency in training data")
    parser.add_argument("--input", required=True, help="Input SimpleQA JSON file")
    parser.add_argument("--output", default="p_true_vs_frequency.png", help="Output plot file")
    parser.add_argument("--save-results", help="Save analysis results to JSON file")
    parser.add_argument("--limit", type=int, help="Limit number of records to process")
    
    args = parser.parse_args()
    
    # Analyze the data
    results = analyze_data(args.input, args.limit)
    
    # Save results if requested
    if args.save_results:
        save_results({"raw_results": results}, args.save_results)
    
    # Plot results using utils
    plot_confidence_vs_frequency(results, args.output)
    
    # Print final summary
    if results:
        print(f"\nProcessed {len(results)} questions")
        avg_p_true = np.mean([r["p_true"] for r in results])
        avg_pre_freq = np.mean([r["pretraining_freq"] for r in results])
        avg_post_freq = np.mean([r["posttraining_freq"] for r in results])
        avg_combined_freq = np.mean([r["combined_freq"] for r in results])
        print(f"Average p_true: {avg_p_true:.4f}")
        print(f"Average pretraining frequency: {avg_pre_freq:.1f}")
        print(f"Average post-training frequency: {avg_post_freq:.1f}")
        print(f"Average combined frequency: {avg_combined_freq:.1f}")


if __name__ == "__main__":
    main() 