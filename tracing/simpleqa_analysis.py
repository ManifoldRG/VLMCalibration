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
import scipy.stats as stats

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

                    You CAN:
                    - Create variations with slightly less specific details if they still clearly refer to the same concept
                    - Reorder or rephrase while maintaining the essential meaning
                    - Add common descriptive words that don't change the core concept

                    DO NOT:
                    - Make the concept completely general or lose its essential meaning
                    - Generate individual words or very short partial phrases
                    - Create variations that could refer to completely different concepts
                    - Remove all specific identifying information

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
    
    # Use API for other indexes (pretraining) - try to get both document and n-gram counts
    else:
        best_result = {
            "doc_count": 0,
            "ngram_count": 0,
            "approx": False,
            "token_ids": [],
            "tokens": [],
            "best_match": text,
            "all_variations": {},
            "semantic_variations": semantic_variations,
            "documents": []
        }
        
        # Try each variation with API - focus on getting document counts when possible
        for variation in all_variations:
            try:
                # First try to get document count using find query type
                find_payload = {"index": index, "query_type": "find", "query": variation}
                find_result = None
                doc_count = 0
                documents = []
                
                try:
                    find_result = post_request(find_payload)
                    if "error" not in find_result:
                        doc_count = int(find_result.get("cnt", 0))
                        print(f"API find for '{variation}': {doc_count} documents")
                        
                        # If we found documents, try to retrieve some samples
                        if doc_count > 0:
                            print(f"Retrieving document samples for '{variation}'...")
                            documents = get_document_samples_from_api(index, find_result, max_docs=3)
                            
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
                        "latency": count_result.get("latency", 0.0),
                        "documents": documents
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
                cnf_documents = []
                find_result = None
                
                try:
                    find_result = post_request(find_payload)
                    if "error" not in find_result:
                        doc_count = int(find_result.get("cnt", 0))
                        
                        # If we found documents, try to retrieve some samples
                        if doc_count > 0:
                            print(f"Retrieving CNF document samples...")
                            cnf_documents = get_document_samples_from_api(index, find_result, max_docs=3)
                            
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
                        best_result["documents"] = cnf_documents
                        
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

def get_document_samples_from_api(index: str, find_result: Dict, max_docs: int = 3, max_disp_len: int = 200) -> List[Dict[str, Any]]:
    """
    Retrieve actual document text samples from API using get_doc_by_rank.
    
    Args:
        index: The index to search in
        find_result: Result from a 'find' query containing segment_by_shard
        max_docs: Maximum number of document samples to retrieve
        max_disp_len: Maximum number of tokens per document
    
    Returns:
        List of document dictionaries with text samples
    """
    documents = []
    doc_count = 0
    
    try:
        segments = find_result.get('segment_by_shard', [])
        if not segments:
            print("No segments found in find_result")
            return documents
            
        for s, (start, end) in enumerate(segments):
            if doc_count >= max_docs:
                break
                
            # Try to get a few documents from this shard
            shard_docs_retrieved = 0
            max_per_shard = min(max_docs - doc_count, 2)  # Limit per shard
            
            for rank in range(start, min(end, start + max_per_shard)):
                if doc_count >= max_docs:
                    break
                    
                try:
                    # Make API call to get document by rank
                    doc_payload = {
                        "index": index,
                        "query_type": "get_doc_by_rank",
                        "s": s,
                        "rank": rank,
                        "max_disp_len": max_disp_len
                    }
                    
                    doc_result = post_request(doc_payload)
                    
                    if "error" not in doc_result:
                        # Extract text from token_ids
                        token_ids = doc_result.get("token_ids", [])
                        text_sample = ""
                        
                        # For API results, we don't have a tokenizer, so use the tokens field if available
                        if "tokens" in doc_result:
                            text_sample = "".join(doc_result["tokens"])
                        elif token_ids:
                            # Fallback: represent as token ID string if we can't decode
                            text_sample = f"[Token IDs: {token_ids[:20]}...]" if len(token_ids) > 20 else f"[Token IDs: {token_ids}]"
                        
                        documents.append({
                            "shard": s,
                            "rank": rank,
                            "doc_ix": doc_result.get("doc_ix"),
                            "doc_len": doc_result.get("doc_len"),
                            "disp_len": doc_result.get("disp_len"),
                            "token_ids": token_ids[:50],  # Limit to first 50 tokens
                            "text_sample": text_sample[:500],  # Limit text length
                            "latency": doc_result.get("latency", 0.0)
                        })
                        doc_count += 1
                        shard_docs_retrieved += 1
                        
                        print(f"Retrieved document from shard {s}, rank {rank}: {len(text_sample)} chars")
                    else:
                        print(f"Error retrieving document shard {s}, rank {rank}: {doc_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    print(f"Exception retrieving document shard {s}, rank {rank}: {e}")
                    continue
                    
                # Add small delay between document retrievals
                time.sleep(0.05)
            
            if doc_count >= max_docs:
                break
                
    except Exception as e:
        print(f"Error in get_document_samples_from_api: {e}")
    
    print(f"Retrieved {len(documents)} document samples from API")
    return documents

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
            documents = []
            
            try:
                find_result = post_request(find_payload)
                if "error" not in find_result:
                    doc_count = int(find_result.get("cnt", 0))
                    print(f"API find result: {doc_count} documents")
                    
                    # If we found documents, try to retrieve some samples
                    if doc_count > 0:
                        print(f"Retrieving document samples...")
                        documents = get_document_samples_from_api(index, find_result, max_docs=5)
                        
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
                    "documents": documents,
                    "error": count_result["error"]
                }
                
            result = {
                "doc_count": doc_count,
                "ngram_count": int(count_result.get("count", 0)),
                "approx": count_result.get("approx", False),
                "token_ids": count_result.get("token_ids", []),
                "tokens": count_result.get("tokens", []),
                "latency": count_result.get("latency", 0.0),
                "documents": documents
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
                "documents": [],
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

def analyze_data(input_file: str) -> Tuple[Dict, List]:
    """Analyze the SimpleQA data and return p_true vs frequency data."""
    print(f"Loading data from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    data = data[:5]
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
            print(f"  {stage}: {count} docs (using: {best_match})")
        
        # Store result
        result = {
            "idx": record.get("idx", i),
            "question": question,
            "concept": concept,
            "p_true": record["p_true"],
            "correct": record["correct"],
            "pretraining_freq": freq_results.get("pretraining", {}).get("doc_count", 0),
            "posttraining_freq": freq_results.get("post_training", {}).get("doc_count", 0),
            "combined_freq": freq_results.get("pretraining", {}).get("doc_count", 0) + freq_results.get("post_training", {}).get("doc_count", 0)
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
        print(f"  p_true: {result['p_true']:.3f}")
        print(f"  pretraining_freq: {result['pretraining_freq']}")
        print(f"  posttraining_freq: {result['posttraining_freq']}")
        print(f"  combined_freq: {result['combined_freq']}")
    
    return {}, results  # Return empty dict for analysis_results since we're not doing binning

def plot_results(analysis_results: Dict, raw_results: List, output_file: str):
    """Plot p_true vs combined frequency (pretraining + post-training)."""
    
    if not raw_results:
        print("No data available for plotting")
        return
    
    # Extract data for plotting
    p_true_values = [r["p_true"] for r in raw_results]
    combined_freqs = [r["combined_freq"] for r in raw_results]
    concepts = [r["concept"] for r in raw_results]
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot p_true vs Combined Frequency
    scatter = ax.scatter(combined_freqs, p_true_values, alpha=0.7, color='purple', s=60)
    ax.set_xlabel('Combined Document Frequency (Pretraining + Post-training)')
    ax.set_ylabel('p_true (Model Confidence)')
    ax.set_title('Model Confidence vs Combined Training Data Frequency')
    ax.grid(True, alpha=0.3)
    
    # Add trend line
    if len(combined_freqs) > 1:
        # Filter out zero frequencies for correlation if needed
        non_zero_combined = [(f, p) for f, p in zip(combined_freqs, p_true_values) if f > 0]
        if non_zero_combined and len(non_zero_combined) > 1:
            freqs_nz, p_true_nz = zip(*non_zero_combined)
            z = np.polyfit(freqs_nz, p_true_nz, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(freqs_nz), max(freqs_nz), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, 
                   label=f'Trend: y={z[0]:.4f}x+{z[1]:.3f}')
            ax.legend()
        # If we have data but all frequencies are zero, fit trend line including zeros
        elif len(set(combined_freqs)) > 1:  # More than one unique frequency value
            z = np.polyfit(combined_freqs, p_true_values, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(combined_freqs), max(combined_freqs), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2,
                   label=f'Trend: y={z[0]:.4f}x+{z[1]:.3f}')
            ax.legend()
    
    # Set log scale if frequencies span wide range
    if max(combined_freqs) > 0 and max(combined_freqs) > 10 * min([f for f in combined_freqs if f > 0], default=1):
        ax.set_xscale('log')
    
    # Add text annotations for interesting points (optional)
    for i, (freq, conf, concept) in enumerate(zip(combined_freqs, p_true_values, concepts)):
        # Annotate points with very high frequency or very high/low confidence
        if freq > max(combined_freqs) * 0.8 or conf > 0.95 or conf < 0.1:
            ax.annotate(concept[:30] + "..." if len(concept) > 30 else concept, 
                       (freq, conf), xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.7)
    
    # Add overall title
    fig.suptitle(f'Model Confidence (p_true) vs Combined Training Data Frequency\n'
                f'Total: {len(raw_results)} questions', fontsize=14, y=0.95)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Confidence vs combined frequency plot saved to {output_file}")
    
    # Print summary statistics
    print(f"\nPlot Summary:")
    print(f"Total questions: {len(raw_results)}")
    print(f"p_true range: [{min(p_true_values):.3f}, {max(p_true_values):.3f}]")
    print(f"Combined frequency range: [{min(combined_freqs)}, {max(combined_freqs)}]")
    
    # Calculate correlation for combined frequency
    if len(set(combined_freqs)) > 1:  # More than one unique frequency value
        corr_combined, p_val_combined = stats.pearsonr(combined_freqs, p_true_values)
        print(f"Combined frequency correlation: r={corr_combined:.4f}, p={p_val_combined:.4f}")
        
        # Also calculate correlation excluding zero frequencies
        non_zero_indices = [i for i, f in enumerate(combined_freqs) if f > 0]
        if len(non_zero_indices) > 1:
            combined_freqs_nz = [combined_freqs[i] for i in non_zero_indices]
            p_true_nz = [p_true_values[i] for i in non_zero_indices]
            corr_nz, p_val_nz = stats.pearsonr(combined_freqs_nz, p_true_nz)
            print(f"Non-zero frequency correlation: r={corr_nz:.4f}, p={p_val_nz:.4f}")
    else:
        print("Not enough frequency variation for correlation analysis")

def main():
    parser = argparse.ArgumentParser(description="Analyze SimpleQA data: plot p_true vs document frequency in training data")
    parser.add_argument("--input", required=True, help="Input SimpleQA JSON file")
    parser.add_argument("--output", default="p_true_vs_frequency.png", help="Output plot file")
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
    plot_results(analysis_results, raw_results, args.output)
    
    # Print summary
    print("\nSummary:")
    print(f"Processed {len(raw_results)} questions")
    if raw_results:
        avg_p_true = np.mean([r["p_true"] for r in raw_results])
        avg_pre_freq = np.mean([r["pretraining_freq"] for r in raw_results])
        avg_post_freq = np.mean([r["posttraining_freq"] for r in raw_results])
        avg_combined_freq = np.mean([r["combined_freq"] for r in raw_results])
        print(f"Average p_true: {avg_p_true:.4f}")
        print(f"Average pretraining frequency: {avg_pre_freq:.1f}")
        print(f"Average post-training frequency: {avg_post_freq:.1f}")
        print(f"Average combined frequency: {avg_combined_freq:.1f}")

if __name__ == "__main__":
    main() 