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
    "post_training": "/mnt/data/post_training_index",  # Post training data (local package)
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
        print(f"Initializing post_training engine... from {OLMO_INDEXES['post_training']}")
        post_training_engine = InfiniGramEngine(
            index_dir=OLMO_INDEXES["post_training"], 
            eos_token_id=tokenizer.eos_token_id
        )

def generate_concept_variations(concept: str) -> List[str]:
    """Generate simple variations of a concept (case variations only)."""
    variations = [concept]
    
    # Add lowercase version
    if concept.lower() != concept:
        variations.append(concept.lower())
    
    # Add title case version
    if concept.title() != concept:
        variations.append(concept.title())
    
    # Add uppercase version for acronyms/abbreviations (only if short)
    if len(concept.split()) <= 2 and concept.upper() != concept:
        variations.append(concept.upper())
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variations = []
    for var in variations:
        if var not in seen:
            seen.add(var)
            unique_variations.append(var)
    
    return unique_variations

def extract_concept_parts(concept: str) -> List[str]:
    """Extract meaningful parts of a concept for partial matching."""
    # REMOVED: This function was causing individual word queries
    # Only return the full concept now
    return [concept]

def generate_semantic_variations_with_gpt4o(concept: str, api_key: str = None) -> List[str]:
    """
    Use GPT-4o to generate semantic variations of a concept for better matching.
    
    Args:
        concept: The original concept
        api_key: OpenAI API key
    
    Returns:
        List of semantic variations that preserve the concept meaning
    """
    try:
        client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": """
                    You are given a concept phrase. Generate 8-10 semantic variations of this EXACT SAME concept that would be likely to appear in text corpora, while preserving the EXACT specificity and meaning.

                    CRITICAL: Keep the concept very specific and the same. Do NOT generalize or make it broader.

                    Think about:
                    - Different phrasings of the EXACT same concept (same specificity level)
                    - Different word orders of the same specific information
                    - Alternative ways to express the same specific details
                    - Preserve ALL key identifying information (names, dates, locations, etc.)
                    - Different grammatical structures for the same concept
                    - How the same specific concept might appear in different text styles

                    DO NOT:
                    - Make the concept more general or broader
                    - Remove specific details like names, dates, or locations
                    - Create variations that could refer to different concepts

                    Return ONLY a JSON list of strings, with no other text. Each variation should be 2-8 words and maintain the same specificity.

                    Example:
                    Input: "eclipse August 28 1802"
                    Output: ["eclipse August 28 1802", "August 28 1802 eclipse", "solar eclipse August 28 1802", "lunar eclipse August 28 1802", "eclipse on August 28 1802", "August 28th 1802 eclipse", "1802 August 28 eclipse", "eclipse of August 28 1802", "28 August 1802 eclipse", "eclipse August 28th 1802"]
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
        
        # Parse the JSON response
        content = response.choices[0].message.content.strip()
        import json
        variations = json.loads(content)
        
        # Ensure we always include the original
        if concept not in variations:
            variations.insert(0, concept)
            
        return variations[:10]  # Limit to 10 variations
        
    except Exception as e:
        print(f"Error generating semantic variations with GPT-4o: {e}")
        # Fallback to simple variations
        return generate_concept_variations(concept)

def count_ngram_with_document_focus(index: str, text: str) -> Dict[str, Any]:
    """Count document occurrences using semantic variations and prioritize document counts over n-gram counts."""
    print(f"Searching for '{text}' in {index} with document-focused analysis")
    
    # Generate semantic variations using GPT-4o
    print("Generating semantic variations with GPT-4o...")
    semantic_variations = generate_semantic_variations_with_gpt4o(text)
    print(f"Generated semantic variations: {semantic_variations}")
    
    # Also generate simple variations as backup
    simple_variations = generate_concept_variations(text)
    
    # Combine and deduplicate
    all_variations = []
    seen = set()
    for var_list in [semantic_variations, simple_variations]:
        for var in var_list:
            if var not in seen:
                seen.add(var)
                all_variations.append(var)
    
    print(f"Total variations to try: {len(all_variations)}")
    
    # Check if this is the post_training index (use local package)
    if index == OLMO_INDEXES["post_training"]:
        try:
            initialize_local_engine()
            
            best_result = {
                "doc_count": 0,
                "ngram_count": 0,
                "approx": False,
                "token_ids": [],
                "tokens": [],
                "documents": [],
                "total_segments": 0,
                "next_token_distribution": None,
                "best_match": text,
                "all_variations": {},
                "semantic_variations": semantic_variations
            }
            
            # First, try each variation individually with document focus
            for variation in all_variations:
                try:
                    input_ids = tokenizer.encode(variation)
                    print(f"Trying variation '{variation}' -> tokens: {input_ids}")
                    
                    # Get both n-gram count and document information
                    count_result = post_training_engine.count(input_ids=input_ids)
                    ngram_count = int(count_result.get("count", 0))
                    
                    doc_count = 0
                    documents = []
                    
                    # If we have n-gram matches, get document count
                    if ngram_count > 0:
                        try:
                            find_result = post_training_engine.find(input_ids=input_ids)
                            # Calculate unique document count from segments
                            total_segments = sum(end - start for start, end in find_result.get('segment_by_shard', []))
                            doc_count = total_segments  # Each segment represents a document match
                            
                            # Get sample documents
                            doc_sample_count = 0
                            max_sample_docs = 3
                            
                            for s, (start, end) in enumerate(find_result.get('segment_by_shard', [])):
                                for rank in range(start, min(end, start + max_sample_docs - doc_sample_count)):
                                    if doc_sample_count >= max_sample_docs:
                                        break
                                    try:
                                        doc = post_training_engine.get_doc_by_rank(s=s, rank=rank, max_disp_len=100)
                                        documents.append({
                                            "shard": s,
                                            "rank": rank,
                                            "doc_ix": doc.get("doc_ix"),
                                            "text_sample": tokenizer.decode(doc.get("token_ids", [])[:50]) if doc.get("token_ids") else ""
                                        })
                                        doc_sample_count += 1
                                    except Exception as e:
                                        print(f"Error getting document: {e}")
                                if doc_sample_count >= max_sample_docs:
                                    break
                                    
                        except Exception as e:
                            print(f"Error finding documents for '{variation}': {e}")
                    
                    var_result = {
                        "doc_count": doc_count,
                        "ngram_count": ngram_count,
                        "approx": count_result.get("approx", False),
                        "token_ids": input_ids,
                        "tokens": [tokenizer.decode([token_id]) for token_id in input_ids],
                        "documents": documents
                    }
                    best_result["all_variations"][variation] = var_result
                    
                    # Prioritize by document count, fallback to n-gram count
                    if doc_count > best_result["doc_count"] or (doc_count == best_result["doc_count"] and ngram_count > best_result["ngram_count"]):
                        best_result.update(var_result)
                        best_result["best_match"] = variation
                        
                except Exception as e:
                    print(f"Error processing variation '{variation}': {e}")
                    continue
            
            # If we have low document matches, try CNF queries with semantic variations
            if best_result["doc_count"] < 10:  # Threshold for trying CNF
                print(f"Low document matches ({best_result['doc_count']}), trying CNF with semantic variations...")
                
                # Try CNF with the top semantic variations
                top_variations = semantic_variations[:5]  # Use top 5 semantic variations
                try:
                    print(f"Creating CNF query with: {top_variations}")
                    cnf_query = [[tokenizer.encode(var) for var in top_variations]]
                    print(f"CNF query token structure: {cnf_query}")
                    
                    # Get CNF document matches
                    cnf_result = post_training_engine.find_cnf(cnf=cnf_query)
                    cnf_doc_count = cnf_result.get("cnt", 0)
                    
                    if cnf_doc_count > best_result["doc_count"]:
                        print(f"CNF query found {cnf_doc_count} document matches (better than {best_result['doc_count']})")
                        
                        # Get CNF n-gram count for reference
                        cnf_count_result = post_training_engine.count_cnf(cnf=cnf_query)
                        cnf_ngram_count = int(cnf_count_result.get("count", 0))
                        
                        best_result["doc_count"] = cnf_doc_count
                        best_result["ngram_count"] = cnf_ngram_count
                        best_result["approx"] = cnf_result.get("approx", False)
                        best_result["best_match"] = f"CNF: {' OR '.join(top_variations)}"
                        best_result["cnf_query"] = True
                        best_result["cnf_variations"] = top_variations
                        
                        # Try to get some documents from CNF
                        try:
                            documents = []
                            doc_count = 0
                            max_docs = 3
                            
                            for s, ptrs in enumerate(cnf_result.get('ptrs_by_shard', [])):
                                for ptr in ptrs[:max_docs - doc_count]:
                                    if doc_count >= max_docs:
                                        break
                                    try:
                                        doc = post_training_engine.get_doc_by_ptr(s=s, ptr=ptr, max_disp_len=200)
                                        documents.append({
                                            "shard": s,
                                            "ptr": ptr,
                                            "doc_ix": doc.get("doc_ix"),
                                            "text_sample": tokenizer.decode(doc.get("token_ids", [])[:100]) if doc.get("token_ids") else ""
                                        })
                                        doc_count += 1
                                    except Exception as e:
                                        print(f"Error getting CNF document: {e}")
                                    if doc_count >= max_docs:
                                        break
                            
                            best_result["documents"] = documents
                            
                        except Exception as e:
                            print(f"Error with CNF find: {e}")
                            
                except Exception as e:
                    print(f"Error with semantic CNF query: {e}")
            
            print(f"Best result for '{text}': {best_result['doc_count']} docs, {best_result['ngram_count']} n-grams with '{best_result.get('best_match', text)}'")
            return best_result
            
        except Exception as e:
            print(f"Exception in local count_ngram_with_document_focus: {e}")
            traceback.print_exc()
            return {
                "doc_count": 0,
                "ngram_count": 0,
                "approx": False,
                "token_ids": [],
                "tokens": [],
                "documents": [],
                "total_segments": 0,
                "best_match": text,
                "error": str(e)
            }
    
    # Use API for other indexes (pretraining) - try to get document information
    else:
        best_result = {
            "doc_count": 0,
            "ngram_count": 0,
            "approx": False,
            "token_ids": [],
            "tokens": [],
            "best_match": text,
            "all_variations": {},
            "semantic_variations": semantic_variations
        }
        
        # Try each variation with API - focus on getting document counts when possible
        for variation in all_variations:
            try:
                # First try to get document count using find query type
                find_payload = {"index": index, "query_type": "find", "query": variation}
                find_result = None
                doc_count = 0
                
                try:
                    find_result = post_request(find_payload)
                    if "error" not in find_result:
                        doc_count = int(find_result.get("cnt", 0))
                        print(f"API find for '{variation}': {doc_count} documents")
                except Exception as e:
                    print(f"API find query failed for '{variation}': {e}")
                    # Fallback to count query
                    find_result = None
                
                # Get n-gram count as well
                count_payload = {"index": index, "query_type": "count", "query": variation}
                count_result = post_request(count_payload)
                ngram_count = 0
                
                if "error" not in count_result:
                    ngram_count = int(count_result.get("count", 0))
                    
                    var_result = {
                        "doc_count": doc_count,
                        "ngram_count": ngram_count,
                        "approx": count_result.get("approx", False),
                        "token_ids": count_result.get("token_ids", []),
                        "tokens": count_result.get("tokens", []),
                        "latency": count_result.get("latency", 0.0)
                    }
                    
                    # If find query worked, use those results
                    if find_result and "error" not in find_result:
                        var_result["find_approx"] = find_result.get("approx", False)
                        var_result["find_latency"] = find_result.get("latency", 0.0)
                    
                    best_result["all_variations"][variation] = var_result
                    
                    # Prioritize by document count, fallback to n-gram count
                    if doc_count > best_result["doc_count"] or (doc_count == best_result["doc_count"] and ngram_count > best_result["ngram_count"]):
                        best_result.update(var_result)
                        best_result["best_match"] = variation
                        
            except Exception as e:
                print(f"Error with API variation '{variation}': {e}")
                continue
            
            time.sleep(0.1)  # Rate limiting
        
        # If still low document matches and we have semantic variations, try CNF query with API
        if best_result["doc_count"] < 10 and len(semantic_variations) > 1:
            try:
                print(f"Trying CNF query with API for low document matches...")
                # Use top semantic variations for CNF
                top_variations = semantic_variations[:4]  # Limit for API
                cnf_query = " OR ".join(top_variations)
                
                # Try find query for documents
                find_payload = {"index": index, "query_type": "find", "query": cnf_query}
                doc_count = 0
                try:
                    find_result = post_request(find_payload)
                    if "error" not in find_result:
                        doc_count = int(find_result.get("cnt", 0))
                except Exception as e:
                    print(f"API CNF find query failed: {e}")
                
                # Get CNF count
                count_payload = {"index": index, "query_type": "count", "query": cnf_query}
                count_result = post_request(count_payload)
                
                if "error" not in count_result:
                    cnf_ngram_count = int(count_result.get("count", 0))
                    
                    if doc_count > best_result["doc_count"] or (doc_count == best_result["doc_count"] and cnf_ngram_count > best_result["ngram_count"]):
                        print(f"API CNF query found {doc_count} documents, {cnf_ngram_count} n-grams (better than {best_result['doc_count']} docs)")
                        best_result["doc_count"] = doc_count
                        best_result["ngram_count"] = cnf_ngram_count
                        best_result["approx"] = count_result.get("approx", False)
                        best_result["token_ids"] = count_result.get("token_ids", [])
                        best_result["tokens"] = count_result.get("tokens", [])
                        best_result["best_match"] = f"CNF: {cnf_query}"
                        best_result["latency"] = count_result.get("latency", 0.0)
                        best_result["cnf_query"] = True
                        best_result["cnf_variations"] = top_variations
                        
            except Exception as e:
                print(f"Error with API CNF query: {e}")
        
        print(f"API best result for '{text}': {best_result['doc_count']} docs, {best_result['ngram_count']} n-grams with '{best_result.get('best_match', text)}'")
        return best_result

def get_frequency_counts_with_document_focus(question: str) -> Dict[str, Any]:
    """Get document-focused frequency counts for a question across different training stages."""
    results = {}
    for stage, index in OLMO_INDEXES.items():
        result = count_ngram_with_document_focus(index, question)
        results[stage] = result
        time.sleep(0.1)  # Rate limiting
    return results

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

def count_ngram(index: str, text: str) -> Dict[str, Any]:
    """Count document occurrences and n-gram occurrences of text in the specified index."""
    
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
            count_result = post_training_engine.count(input_ids=input_ids)
            print(f"Local engine count result: {count_result}")
            
            # Get comprehensive information with document focus
            result = {
                "doc_count": 0,
                "ngram_count": int(count_result.get("count", 0)),
                "approx": count_result.get("approx", False),
                "token_ids": input_ids,
                "tokens": [tokenizer.decode([token_id]) for token_id in input_ids],
            }
            
            # If we found n-gram matches, get document information
            if result["ngram_count"] > 0:
                try:
                    # Find documents containing this text
                    find_result = post_training_engine.find(input_ids=input_ids)
                    print(f"Find result: {find_result}")
                    
                    # Calculate document count from segments
                    total_segments = sum(end - start for start, end in find_result.get('segment_by_shard', []))
                    result["doc_count"] = total_segments
                    
                    # Get a sample of documents (up to 5)
                    documents = []
                    doc_count = 0
                    max_docs = 5
                    
                    for s, (start, end) in enumerate(find_result.get('segment_by_shard', [])):
                        for rank in range(start, min(end, start + max_docs - doc_count)):
                            if doc_count >= max_docs:
                                break
                            try:
                                doc = post_training_engine.get_doc_by_rank(s=s, rank=rank, max_disp_len=100)
                                documents.append({
                                    "shard": s,
                                    "rank": rank,
                                    "doc_ix": doc.get("doc_ix"),
                                    "doc_len": doc.get("doc_len"),
                                    "disp_len": doc.get("disp_len"),
                                    "token_ids": doc.get("token_ids", [])[:50],  # Limit to first 50 tokens
                                    "text_sample": tokenizer.decode(doc.get("token_ids", [])[:50]) if doc.get("token_ids") else ""
                                })
                                doc_count += 1
                            except Exception as e:
                                print(f"Error getting document at shard {s}, rank {rank}: {e}")
                        if doc_count >= max_docs:
                            break
                    
                    result["documents"] = documents
                    result["total_segments"] = total_segments
                    
                    # Get next-token distribution for additional analysis
                    try:
                        ntd_result = post_training_engine.ntd(prompt_ids=input_ids[:-1], max_support=10)
                        result["next_token_distribution"] = {
                            "prompt_cnt": ntd_result.get("prompt_cnt", 0),
                            "top_continuations": [
                                {
                                    "token_id": token_id,
                                    "token": tokenizer.decode([token_id]),
                                    "count": info.get("cont_cnt", 0),
                                    "prob": info.get("prob", 0.0)
                                }
                                for token_id, info in sorted(
                                    ntd_result.get("result_by_token_id", {}).items(),
                                    key=lambda x: x[1].get("cont_cnt", 0),
                                    reverse=True
                                )[:10]  # Top 10 continuations
                            ],
                            "approx": ntd_result.get("approx", False)
                        }
                    except Exception as e:
                        print(f"Error getting next-token distribution: {e}")
                        result["next_token_distribution"] = None
                        
                except Exception as e:
                    print(f"Error finding documents: {e}")
                    result["documents"] = []
                    result["total_segments"] = 0
            else:
                result["documents"] = []
                result["total_segments"] = 0
                result["next_token_distribution"] = None
            
            print(f"Comprehensive local engine result: {result}")
            return result
            
        except Exception as e:
            print(f"Exception in local count_ngram: {e}")
            traceback.print_exc()
            return {
                "doc_count": 0,
                "ngram_count": 0,
                "approx": False,
                "token_ids": [],
                "tokens": [],
                "documents": [],
                "total_segments": 0,
                "next_token_distribution": None,
                "error": str(e)
            }
    
    # Use API for other indexes (pretraining) - try to get both document and n-gram counts
    else:
        try:
            # Try to get document count first
            find_payload = {"index": index, "query_type": "find", "query": text}
            doc_count = 0
            find_result = None
            
            try:
                find_result = post_request(find_payload)
                if "error" not in find_result:
                    doc_count = int(find_result.get("cnt", 0))
                    print(f"API find result: {doc_count} documents")
            except Exception as e:
                print(f"API find query failed: {e}")
            
            # Get n-gram count
            count_payload = {"index": index, "query_type": "count", "query": text}
            count_result = post_request(count_payload)
            print(f"API Count Payload: {count_payload}")
            print(f"API Count Result: {count_result}")
            
            # Check for error key first as per API documentation
            if "error" in count_result:
                print(f"API Error: {count_result['error']}")
                return {
                    "doc_count": doc_count,
                    "ngram_count": 0,
                    "approx": False,
                    "token_ids": [],
                    "tokens": [],
                    "error": count_result["error"]
                }
                
            result = {
                "doc_count": doc_count,
                "ngram_count": int(count_result.get("count", 0)),
                "approx": count_result.get("approx", False),
                "token_ids": count_result.get("token_ids", []),
                "tokens": count_result.get("tokens", []),
                "latency": count_result.get("latency", 0.0)
            }
            
            # If find query worked, add those details
            if find_result and "error" not in find_result:
                result["find_approx"] = find_result.get("approx", False)
                result["find_latency"] = find_result.get("latency", 0.0)
            
            return result
        except Exception as e:
            print(f"Exception in API count_ngram: {e}")
            return {
                "doc_count": 0,
                "ngram_count": 0,
                "approx": False,
                "token_ids": [],
                "tokens": [],
                "error": str(e)
            }

def get_frequency_counts(question: str) -> Dict[str, Any]:
    """Get frequency counts and comprehensive information for a question across different training stages."""
    results = {}
    for stage, index in OLMO_INDEXES.items():
        result = count_ngram(index, question)
        results[stage] = result
        time.sleep(0.1)  # Rate limiting
    return results

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
    
    data = data[:1]
    # Process all records in the dataset
    print(f"Processing {len(data)} records from {input_file}")
    
    # Extract and process each record
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
        
        # Get frequency counts with improved matching
        print(f"Searching for frequency counts...")
        freq_results = get_frequency_counts_with_document_focus(concept)
        
        # Print detailed results for debugging
        print(f"\nFrequency Results:")
        for stage, result in freq_results.items():
            count = result.get("doc_count", 0)
            best_match = result.get("best_match", concept)
            all_variations = result.get("all_variations", {})
            
            print(f"  {stage}: {count} docs")
            print(f"    Best match: '{best_match}'")
            if all_variations:
                print(f"    All variations tried:")
                for var, var_result in all_variations.items():
                    var_count = var_result.get("doc_count", 0)
                    print(f"      '{var}': {var_count} docs")
            
            # If we have documents, show a sample
            documents = result.get("documents", [])
            if documents:
                print(f"    Sample documents ({len(documents)} shown):")
                for j, doc in enumerate(documents[:2]):  # Show first 2
                    sample = doc.get("text_sample", "")[:100] + "..." if len(doc.get("text_sample", "")) > 100 else doc.get("text_sample", "")
                    print(f"      {j+1}. {sample}")
        
        print(f"\nFinal frequency counts:")
        for stage, result in freq_results.items():
            count = result.get("doc_count", 0)
            match_used = result.get("best_match", concept)
            print(f"  {stage}: {count} docs (using: {match_used})")
        
        # Store result
        result = {
            "idx": record.get("idx", i),
            "question": question,
            "concept": concept,
            "p_true": record["p_true"],
            "correct": record["correct"],
            "frequency_results": freq_results,
            "frequency_counts": {stage: data.get("doc_count", 0) for stage, data in freq_results.items()}
        }
        results.append(result)
        
        # Add some delay to avoid hitting rate limits
        time.sleep(0.3)
        
        # Show progress
        print(f"Completed {i+1}/{len(data)} records")
    
    print("\n" + "="*60)
    print("SUMMARY OF ALL RESULTS:")
    print("="*60)
    
    for i, result in enumerate(results):
        print(f"Record {i+1}: {result['concept']}")
        for stage, count in result['frequency_counts'].items():
            # Also show n-gram counts for reference
            freq_result = result['frequency_results'][stage]
            ngram_count = freq_result.get('ngram_count', 0)
            print(f"  {stage}: {count} docs ({ngram_count} n-grams)")
    
    # Now proceed to binning and ECE calculation
    print(f"\nProceeding to document frequency binning and ECE calculation...")
    
    # Create frequency bins for each training stage based on document counts
    analysis_results = {}
    
    for stage in OLMO_INDEXES.keys():
        print(f"\nProcessing {stage} stage...")
        
        # Extract data for this stage using document counts
        stage_data = []
        for result in results:
            doc_count = result['frequency_counts'].get(stage, 0)
            if doc_count > 0:  # Only include non-zero document frequencies
                stage_data.append({
                    'frequency': doc_count,
                    'p_true': result['p_true'],
                    'correct': result['correct']
                })
        
        if not stage_data:
            print(f"  No data with non-zero document frequencies for {stage}")
            analysis_results[stage] = []
            continue
        
        print(f"  {len(stage_data)} samples with non-zero document frequencies")
        
        # Sort by frequency for binning
        stage_data.sort(key=lambda x: x['frequency'])
        
        # Create frequency bins (logarithmic binning for better distribution)
        frequencies = [d['frequency'] for d in stage_data]
        min_freq = min(frequencies)
        max_freq = max(frequencies)
        
        if min_freq == max_freq:
            # All frequencies are the same, create single bin
            n_bins = 1
            bin_edges = [min_freq - 0.5, max_freq + 0.5]
        else:
            # Use 5 bins or fewer if we don't have enough data
            n_bins = min(5, len(stage_data) // 2)  # At least 2 samples per bin
            n_bins = max(1, n_bins)  # At least 1 bin
            
            # Create logarithmic bins
            if min_freq > 0:
                bin_edges = np.logspace(np.log10(min_freq), np.log10(max_freq), n_bins + 1)
            else:
                # Handle case where min_freq is 0
                bin_edges = np.linspace(min_freq, max_freq, n_bins + 1)
        
        print(f"  Creating {n_bins} bins with edges: {bin_edges}")
        
        # Assign data to bins and calculate ECE for each bin
        bin_results = []
        
        for i in range(n_bins):
            bin_lower = bin_edges[i]
            bin_upper = bin_edges[i + 1]
            
            # Find samples in this bin
            bin_samples = []
            for sample in stage_data:
                freq = sample['frequency']
                if i == n_bins - 1:  # Last bin includes upper edge
                    if bin_lower <= freq <= bin_upper:
                        bin_samples.append(sample)
                else:
                    if bin_lower <= freq < bin_upper:
                        bin_samples.append(sample)
            
            if not bin_samples:
                continue
            
            # Calculate statistics for this bin
            bin_frequencies = [s['frequency'] for s in bin_samples]
            bin_confidences = np.array([s['p_true'] for s in bin_samples])
            bin_correct = np.array([s['correct'] for s in bin_samples])
            
            # Calculate ECE for this bin
            if len(bin_samples) > 1:
                bin_ece = calculate_ece(bin_confidences, bin_correct, n_bins=min(5, len(bin_samples)))
            else:
                # For single sample, ECE is just |confidence - accuracy|
                bin_ece = abs(bin_confidences[0] - bin_correct[0])
            
            avg_frequency = np.mean(bin_frequencies)
            avg_confidence = np.mean(bin_confidences)
            accuracy = np.mean(bin_correct)
            
            bin_result = {
                "bin_id": i,
                "frequency_range": [bin_lower, bin_upper],
                "avg_frequency": avg_frequency,
                "count": len(bin_samples),
                "ece": bin_ece,
                "avg_confidence": avg_confidence,
                "accuracy": accuracy,
                "calibration_error": abs(avg_confidence - accuracy)
            }
            
            bin_results.append(bin_result)
            
            print(f"    Bin {i+1}: doc_freq=[{bin_lower:.1f}, {bin_upper:.1f}], "
                  f"n={len(bin_samples)}, avg_doc_freq={avg_frequency:.1f}, "
                  f"ECE={bin_ece:.4f}, conf={avg_confidence:.3f}, acc={accuracy:.3f}")
        
        analysis_results[stage] = bin_results
    
    return analysis_results, results

def plot_results(analysis_results: Dict, output_file: str):
    """Plot ECE histogram from p_true with frequency bins for pretraining and posttraining."""
    
    # Prepare data for histogram
    pretraining_data = analysis_results.get('pretraining', [])
    posttraining_data = analysis_results.get('post_training', [])
    
    if not pretraining_data and not posttraining_data:
        print("No data available for plotting")
        return
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: ECE Histogram for Pretraining
    if pretraining_data:
        ece_values = [d["ece"] for d in pretraining_data]
        frequencies = [d["avg_frequency"] for d in pretraining_data]
        counts = [d["count"] for d in pretraining_data]
        
        # Create histogram bars
        bins = range(len(ece_values))
        bars1 = ax1.bar(bins, ece_values, color='blue', alpha=0.7, 
                       label=f'Pretraining ({len(ece_values)} bins)')
        
        # Add frequency labels on x-axis
        ax1.set_xticks(bins)
        ax1.set_xticklabels([f'{int(f)}' for f in frequencies], rotation=45, ha='right')
        ax1.set_xlabel('Document Frequency in Training Data')
        ax1.set_ylabel('Expected Calibration Error (ECE)')
        ax1.set_title('ECE from p_true - Pretraining Data')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on top of bars
        for i, (bar, count, ece) in enumerate(zip(bars1, counts, ece_values)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'n={count}\nECE={ece:.3f}', 
                    ha='center', va='bottom', fontsize=8)
    else:
        ax1.text(0.5, 0.5, 'No pretraining data available', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('ECE from p_true - Pretraining Data')
    
    # Plot 2: ECE Histogram for Post-training
    if posttraining_data:
        ece_values = [d["ece"] for d in posttraining_data]
        frequencies = [d["avg_frequency"] for d in posttraining_data]
        counts = [d["count"] for d in posttraining_data]
        
        # Create histogram bars
        bins = range(len(ece_values))
        bars2 = ax2.bar(bins, ece_values, color='orange', alpha=0.7,
                       label=f'Post-training ({len(ece_values)} bins)')
        
        # Add frequency labels on x-axis
        ax2.set_xticks(bins)
        ax2.set_xticklabels([f'{int(f)}' for f in frequencies], rotation=45, ha='right')
        ax2.set_xlabel('Document Frequency in Training Data')
        ax2.set_ylabel('Expected Calibration Error (ECE)')
        ax2.set_title('ECE from p_true - Post-training Data')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add count labels on top of bars
        for i, (bar, count, ece) in enumerate(zip(bars2, counts, ece_values)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'n={count}\nECE={ece:.3f}', 
                    ha='center', va='bottom', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'No post-training data available', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('ECE from p_true - Post-training Data')
    
    # Add overall title
    total_bins = len(pretraining_data) + len(posttraining_data)
    fig.suptitle(f'ECE (from p_true) Histogram by Document Frequency\n'
                f'Total: {total_bins} frequency bins', fontsize=14, y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"ECE histogram saved to {output_file}")
    
    # Print summary statistics
    print(f"\nHistogram Summary:")
    if pretraining_data:
        pre_avg_ece = np.mean([d["ece"] for d in pretraining_data])
        pre_total_samples = sum([d["count"] for d in pretraining_data])
        print(f"Pretraining: {len(pretraining_data)} bins, avg ECE = {pre_avg_ece:.4f}, total samples = {pre_total_samples}")
    
    if posttraining_data:
        post_avg_ece = np.mean([d["ece"] for d in posttraining_data])
        post_total_samples = sum([d["count"] for d in posttraining_data])
        print(f"Post-training: {len(posttraining_data)} bins, avg ECE = {post_avg_ece:.4f}, total samples = {post_total_samples}")

def main():
    parser = argparse.ArgumentParser(description="Analyze SimpleQA data with infini-gram document frequency analysis")
    parser.add_argument("--input", required=True, help="Input SimpleQA JSON file")
    parser.add_argument("--output", default="ece_vs_document_frequency.png", help="Output plot file")
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