"""
infini_gram_api.py

Functions for querying the Infini-gram API for pretraining corpus data.
"""

import time
from typing import Dict, Any, List
import requests
import traceback

API_URL = "https://api.infini-gram.io/"

# OLMO index for pretraining data
PRETRAINING_INDEX = "v4_olmo-2-1124-13b-instruct_llama"


def post_request(payload: Dict[str, Any], retries: int = 3, backoff: float = 0.5) -> Dict:
    """POST request with retry logic."""
    for attempt in range(retries):
        try:
            r = requests.post(API_URL, json=payload, timeout=30)
            if not r.ok:
                print(f"DEBUG - HTTP {r.status_code} error response: {r.text}")
                print(f"DEBUG - payload: {payload}")
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                print(f"DEBUG - payload: {payload}")
                print(f"DEBUG - API error response: {data}")
                raise RuntimeError(data["error"])
            return data
        except Exception as e:
            print(f"DEBUG - Request attempt {attempt + 1} failed: {type(e).__name__}: {str(e)}")
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (2 ** attempt))
    return {}


def query_pretraining_api(concept: str, semantic_variations: List[str], zero_points: bool) -> Dict[str, Any]:
    """Query pretraining corpus via API with concept and semantic variations."""
    index = PRETRAINING_INDEX
    result = {
        "variations_contexts": {},   # Store document contexts for each variation
        "all_variations_data": {}    # Store all data for each variation
    }
    
    # Ensure we always have at least the concept itself to try
    variations_to_try = semantic_variations.copy() if semantic_variations else []
    if concept not in variations_to_try:
        variations_to_try.insert(0, concept)
    
    val = 0 if zero_points else 1 # check for zero points to be included
    # Try each variation
    for variation in variations_to_try:
        try:
            # Get document count
            find_payload = {"index": index, "query_type": "find", "query": variation}
            find_result = post_request(find_payload)
            if "error" not in find_result:
                doc_count = int(find_result.get("cnt", 0))
                
                # Get n-gram count
                count_payload = {"index": index, "query_type": "count", "query": variation}
                count_result = post_request(count_payload)
                
                if "error" not in count_result:
                    ngram_count = int(count_result.get("count", 0))
                    
                    # Store data for this variation
                    if doc_count >= val or ngram_count >= val:
                        result["all_variations_data"][variation] = {
                            "doc_count": doc_count,
                            "ngram_count": ngram_count,
                            "tokens": "".join(count_result.get("tokens", []))
                        }
                    
                    # Get document contexts for this variation if docs exist
                    if doc_count >= val and "segment_by_shard" in find_result:
                        document_contexts = []
                        segment_by_shard = find_result.get("segment_by_shard", [])
                        
                        samples_collected = 0
                        for shard_idx, (start, end) in enumerate(segment_by_shard):
                            if samples_collected >= 5:
                                break
                            
                            # Get up to 5 documents from this shard
                            sample_count = min(5, end - start)
                            for rank in range(start, start + sample_count):
                                doc_payload = {
                                    "index": index,
                                    "query_type": "get_doc_by_rank",
                                    "query": variation,
                                    "s": shard_idx,
                                    "rank": rank,
                                    "max_disp_len": 150
                                }
                                
                                try:
                                    doc_result = post_request(doc_payload)
                                    if "error" not in doc_result and doc_result:
                                        # Extract relevant document data
                                        document_contexts.append({
                                            "text": "".join(doc_result.get("tokens", [])),
                                            "spans": doc_result.get("spans", []),
                                            "metadata": doc_result.get("metadata", "")
                                        })
                                        samples_collected += 1
                                        
                                        if samples_collected >= 5:
                                            break
                                except Exception as e:
                                    print(f"Error getting simple query document: {e}")
                        
                        # Store contexts for this variation
                        result["variations_contexts"][variation] = document_contexts
                        
                        if doc_count >= val or ngram_count >= val:
                            # Also store document contexts in the all_variations_data dictionary
                            result["all_variations_data"][variation]["document_contexts"] = document_contexts
            
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            print(f"Error with variation '{variation}': {e}")
            continue
    
    # Try CNF query if results are low for individual variations
    total_docs = sum(data.get("doc_count", 0) for data in result["all_variations_data"].values())
    if total_docs < 10 and len(variations_to_try) > 1:
        try:
            cnf_query = " AND ".join(variations_to_try[:4])  # API limit: max 4 terms
            
            # Try CNF find
            find_payload = {"index": index, "query_type": "find_cnf", "query": cnf_query}
            find_result = post_request(find_payload)
            
            if "error" not in find_result:
                cnf_doc_count = int(find_result.get("cnt", 0))
                
                # Get CNF count
                count_payload = {"index": index, "query_type": "count_cnf", "query": cnf_query}
                count_result = post_request(count_payload)
                
                if "error" not in count_result:
                    cnf_ngram_count = int(count_result.get("count", 0))
                    cnf_key = f"CNF: {cnf_query}"
                    
                    # Store data for CNF query
                    if cnf_doc_count >= val or cnf_ngram_count >= val:
                        result["all_variations_data"][cnf_key] = {
                            "doc_count": cnf_doc_count,
                            "ngram_count": cnf_ngram_count,
                            "tokens": "".join(count_result.get("tokens", []))
                        }
                        
                        # Get document samples for CNF query
                        if cnf_doc_count >= val and "ptrs_by_shard" in find_result:
                            document_contexts = []
                            ptrs_by_shard = find_result.get("ptrs_by_shard", [])
                            
                            samples_collected = 0
                            for shard_idx, ptrs in enumerate(ptrs_by_shard):
                                if samples_collected >= 5 or not ptrs:
                                    break
                                    
                                # Get up to 3 documents from this shard
                                for ptr in ptrs[:3]:
                                    doc_payload = {
                                        "index": index,
                                        "query_type": "get_doc_by_ptr",
                                        "query": cnf_query,
                                        "s": shard_idx,
                                        "ptr": ptr,
                                        "max_disp_len": 150
                                    }
                                    
                                    try:
                                        doc_result = post_request(doc_payload)
                                        if "error" not in doc_result and doc_result:
                                            # Extract relevant document data
                                            document_contexts.append({
                                                "text": "".join(doc_result.get("tokens", [])),
                                                "spans": doc_result.get("spans", []),
                                                "metadata": doc_result.get("metadata", "")
                                            })
                                            samples_collected += 1
                                            
                                            if samples_collected >= 5:
                                                break
                                    except Exception as e:
                                        print(f"Error getting CNF document: {e}")
                            
                            # Store CNF results
                            result["variations_contexts"][cnf_key] = document_contexts
                            result["all_variations_data"][cnf_key]["document_contexts"] = document_contexts
        except Exception as e:
            print(f"Error with CNF query: {e}")
    
    # Calculate total statistics for reporting
    total_variations = len(result["all_variations_data"])
    total_contexts = sum(len(contexts) for contexts in result["variations_contexts"].values())
    total_doc_count = sum(data.get("doc_count", 0) for data in result["all_variations_data"].values())
    total_ngram_count = sum(data.get("ngram_count", 0) for data in result["all_variations_data"].values())
    
    print(f"Pretraining API: found {total_doc_count} docs across all variations")
    print(f"  Retrieved contexts for {len(result['variations_contexts'])} variations")
    print(f"  Collected data for {total_variations} variations")
    print(f"  Total contexts: {total_contexts}")
    
    # Add total counts to result
    result["total_doc_count"] = total_doc_count
    result["total_ngram_count"] = total_ngram_count
    
    return result 