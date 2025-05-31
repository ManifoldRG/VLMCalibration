"""
infini_gram_local.py

Functions for querying the local Infini-gram package for post-training corpus data.
"""

from typing import Dict, Any, List
from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer

# OLMO index for post-training data
POST_TRAINING_INDEX = "/mnt/data/post_training_index"

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
        print(f"Initializing post_training engine from {POST_TRAINING_INDEX}")
        post_training_engine = InfiniGramEngine(
            index_dir=POST_TRAINING_INDEX, 
            eos_token_id=tokenizer.eos_token_id
        )


def query_post_training_local(concept: str, semantic_variations: List[str]) -> Dict[str, Any]:
    """Query post-training corpus via local engine with concept and semantic variations."""
    initialize_local_engine()
    
    best_result = {
        "doc_count": 0,
        "ngram_count": 0,
        "tokens": "",
        "documents": [],
        "best_match": concept,
        "document_contexts": []
    }
    
    # Try each variation
    for variation in semantic_variations:
        try:
            input_ids = tokenizer.encode(variation)
            
            # Get n-gram count
            count_result = post_training_engine.count(input_ids=input_ids)
            ngram_count = int(count_result.get("count", 0))
            
            if ngram_count > 0:
                # Get document count
                find_result = post_training_engine.find(input_ids=input_ids)
                segments = find_result.get("segment_by_shard", [])
                
                doc_count = 0
                for segment in segments:
                    if isinstance(segment, list) and len(segment) == 2:
                        start, end = segment
                        doc_count += (end - start)
                
                # Update best result if this variation is better
                if doc_count > best_result["doc_count"] or (doc_count == best_result["doc_count"] and ngram_count > best_result["ngram_count"]):
                    best_result["doc_count"] = doc_count
                    best_result["ngram_count"] = ngram_count
                    best_result["best_match"] = variation
                    best_result["tokens"] = tokenizer.decode(input_ids)
                    
                    # Get document samples with proper context
                    document_contexts = []
                    samples_collected = 0
                    
                    for shard_idx, segment in enumerate(segments):
                        if samples_collected >= 5:
                            break
                            
                        if isinstance(segment, list) and len(segment) == 2:
                            start, end = segment
                            if start >= 0 and end > start:
                                # Get up to 3 documents from this shard
                                for rank_offset in range(min(3, end - start)):
                                    try:
                                        doc = post_training_engine.get_doc_by_rank(
                                            s=shard_idx, 
                                            rank=start + rank_offset, 
                                            max_disp_len=150  # Get ~150 tokens for context
                                        )
                                        
                                        token_ids = doc.get("token_ids", [])
                                        if token_ids:
                                            text_context = tokenizer.decode(token_ids)
                                            
                                            document_contexts.append({
                                                "shard": shard_idx,
                                                "rank": start + rank_offset,
                                                "text": text_context,
                                                "doc_ix": doc.get("doc_ix"),
                                                "doc_len": doc.get("doc_len")
                                            })
                                            samples_collected += 1
                                            
                                            if samples_collected >= 5:
                                                break
                                    except Exception as e:
                                        print(f"Error getting document: {e}")
                    
                    best_result["document_contexts"] = document_contexts
                    
        except Exception as e:
            print(f"Error with variation '{variation}': {e}")
            continue
    
    # Try CNF query if results are low
    if best_result["doc_count"] < 10 and len(semantic_variations) > 1:
        try:
            # Create CNF query - each variation is a clause (OR between them)
            cnf = [[tokenizer.encode(var)] for var in semantic_variations[:4]]  # Max 4 variations
            
            # Get CNF document count
            cnf_result = post_training_engine.find_cnf(cnf=cnf)
            cnf_doc_count = cnf_result.get("cnt", 0)
            
            if cnf_doc_count > best_result["doc_count"]:
                # Get CNF n-gram count
                cnf_count_result = post_training_engine.count_cnf(cnf=cnf)
                cnf_ngram_count = int(cnf_count_result.get("count", 0))
                
                best_result["doc_count"] = cnf_doc_count
                best_result["ngram_count"] = cnf_ngram_count
                best_result["best_match"] = f"CNF: {' OR '.join(semantic_variations[:4])}"
                
                # Get document samples for CNF query
                document_contexts = []
                ptrs_by_shard = cnf_result.get("ptrs_by_shard", [])
                samples_collected = 0
                
                for shard_idx, ptrs in enumerate(ptrs_by_shard):
                    if samples_collected >= 5:
                        break
                        
                    # Get up to 3 documents from this shard
                    for ptr in ptrs[:3]:
                        try:
                            # For CNF queries, use get_doc_by_ptr instead of get_doc_by_rank
                            doc = post_training_engine.get_doc_by_ptr(
                                s=shard_idx,
                                ptr=ptr,
                                max_disp_len=150  # Get ~150 tokens for context
                            )
                            
                            token_ids = doc.get("token_ids", [])
                            if token_ids:
                                text_context = tokenizer.decode(token_ids)
                                
                                document_contexts.append({
                                    "shard": shard_idx,
                                    "ptr": ptr,
                                    "text": text_context,
                                    "doc_ix": doc.get("doc_ix"),
                                    "doc_len": doc.get("doc_len")
                                })
                                samples_collected += 1
                                
                                if samples_collected >= 5:
                                    break
                        except Exception as e:
                            print(f"Error getting CNF document: {e}")
                
                best_result["document_contexts"] = document_contexts
        except Exception as e:
            print(f"Error with CNF query: {e}")
    
    print(f"Post-training local: {best_result['doc_count']} docs, {best_result['ngram_count']} n-grams with '{best_result['best_match']}'")
    print(f"  Retrieved {len(best_result['document_contexts'])} document contexts")
    return best_result 