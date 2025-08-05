"""
infini_gram_local.py

Functions for querying the local Infini-gram package for post-training corpus data.
"""

from typing import Dict, Any, List
from infini_gram.engine import InfiniGramEngine
from transformers import AutoTokenizer

# OLMO index for post-training data
POST_TRAINING_INDEX = "/mnt/data/post_training_index"


def create_tokenizer():
    """Create and return a new tokenizer instance."""
    print("Creating tokenizer...")
    return AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", 
        add_bos_token=False, 
        add_eos_token=False
    )


def create_engine(tokenizer):
    """Create and return a new engine instance."""
    print(f"Creating post_training engine from {POST_TRAINING_INDEX}")
    return InfiniGramEngine(
        index_dir=POST_TRAINING_INDEX, 
        eos_token_id=tokenizer.eos_token_id
    )


def query_post_training_local(concept: str, semantic_variations: List[str]) -> Dict[str, Any]:
    """Query post-training corpus via local engine with concept and semantic variations."""
    # Create fresh instances for this query
    tokenizer = create_tokenizer()
    engine = create_engine(tokenizer)
    
    # Ensure we always have at least the concept itself to try
    variations_to_try = semantic_variations.copy() if semantic_variations else []
    if concept not in variations_to_try:
        variations_to_try.insert(0, concept)
    
    print(f"PAYLOAD: concept={concept}, variations={variations_to_try}")
    
    result = {
        "variations_contexts": {},   # Store document contexts for each variation
        "all_variations_data": {}    # Store all data for each variation
    }
    
    # Try each variation
    for variation in variations_to_try:
        try:
            input_ids = tokenizer.encode(variation)
            
            # Get n-gram count
            count_result = engine.count(input_ids=input_ids)
            print(f"COUNT PAYLOAD: input_ids={input_ids}")
            print(f"COUNT RESULT: {count_result}")
            
            ngram_count = int(count_result.get("count", 0))
            
            if ngram_count > 0:
                # Get document count
                print(f"FIND PAYLOAD: input_ids={input_ids}")
                find_result = engine.find(input_ids=input_ids)
                print(f"FIND RESULT: {find_result}")
                
                segments = find_result.get("segment_by_shard", [])
                
                doc_count = 0
                for segment in segments:
                    # segment_by_shard returns a list of 2-tuples; handle both tuple and list for safety
                    if isinstance(segment, (list, tuple)) and len(segment) == 2:
                        start, end = segment
                        doc_count += (end - start)
                
                # Store data for this variation
                result["all_variations_data"][variation] = {
                    "doc_count": doc_count,
                    "ngram_count": ngram_count,
                    "tokens": tokenizer.decode(input_ids)
                }
                
                # Get document samples with proper context
                document_contexts = []
                samples_collected = 0
                
                for shard_idx, segment in enumerate(segments):
                    if samples_collected >= 5:
                        break
                        
                    # Accept both tuple and list since the engine may return either
                    if isinstance(segment, (list, tuple)) and len(segment) == 2:
                        start, end = segment
                        if start >= 0 and end > start:
                            # Get up to 3 documents from this shard
                            for rank_offset in range(min(3, end - start)):
                                try:
                                    print(f"GET_DOC_BY_RANK PAYLOAD: shard={shard_idx}, rank={start + rank_offset}, max_disp_len=150")
                                    doc = engine.get_doc_by_rank(
                                        s=shard_idx, 
                                        rank=start + rank_offset, 
                                        max_disp_len=150  # Get ~150 tokens for context
                                    )
                                    print(f"GET_DOC_BY_RANK RESULT: {doc}")
                                    
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
                
                # Store contexts for this variation
                result["variations_contexts"][variation] = document_contexts
                result["all_variations_data"][variation]["document_contexts"] = document_contexts
            else:
                # Store data even for zero results
                result["all_variations_data"][variation] = {
                    "doc_count": 0,
                    "ngram_count": ngram_count,
                    "tokens": tokenizer.decode(input_ids)
                }
                result["variations_contexts"][variation] = []
                result["all_variations_data"][variation]["document_contexts"] = []
                    
        except Exception as e:
            print(f"Error with variation '{variation}': {e}")
            continue
    
    # Try CNF query if results are low
    total_docs = sum(data.get("doc_count", 0) for data in result["all_variations_data"].values())
    if total_docs < 10 and len(variations_to_try) > 1:
        try:
            # Create CNF query - each variation is a clause (AND between them)
            cnf = [[tokenizer.encode(var)] for var in variations_to_try[:4]]  # Max 4 variations
            
            # Get CNF document count
            print(f"FIND_CNF PAYLOAD: cnf={cnf}")
            cnf_result = engine.find_cnf(cnf=cnf)
            print(f"FIND_CNF RESULT: {cnf_result}")
            
            cnf_doc_count = cnf_result.get("cnt", 0)
            
            if cnf_doc_count > 0:
                # Get CNF n-gram count
                print(f"COUNT_CNF PAYLOAD: cnf={cnf}")
                cnf_count_result = engine.count_cnf(cnf=cnf)
                print(f"COUNT_CNF RESULT: {cnf_count_result}")
                
                cnf_ngram_count = int(cnf_count_result.get("count", 0))
                cnf_key = f"CNF: {' AND '.join(variations_to_try[:4])}"
                
                # Store data for CNF query
                result["all_variations_data"][cnf_key] = {
                    "doc_count": cnf_doc_count,
                    "ngram_count": cnf_ngram_count,
                    "tokens": ""  # CNF doesn't have simple token representation
                }
                
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
                            print(f"GET_DOC_BY_PTR PAYLOAD: shard={shard_idx}, ptr={ptr}, max_disp_len=150")
                            # For CNF queries, use get_doc_by_ptr instead of get_doc_by_rank
                            doc = engine.get_doc_by_ptr(
                                s=shard_idx,
                                ptr=ptr,
                                max_disp_len=150  # Get ~150 tokens for context
                            )
                            print(f"GET_DOC_BY_PTR RESULT: {doc}")
                            
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
    
    print(f"Post-training local: found {total_doc_count} docs across all variations")
    print(f"  Retrieved contexts for {len(result['variations_contexts'])} variations")
    print(f"  Collected data for {total_variations} variations")
    print(f"  Total contexts: {total_contexts}")
    
    # Add total counts to result
    result["total_doc_count"] = total_doc_count
    result["total_ngram_count"] = total_ngram_count
    
    print(f"FINAL RESULT: {result}")
    return result


if __name__ == "__main__":
    import sys
    
    # Test the local post-training query functionality
    print("Testing local post-training index query...")
    from huggingface_hub import login
    login(token='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')
    # Default test concept and variations
    test_concept = "Jamia Millia"
    test_variations = [
        "natural language processing",
        "NLP",
        "language processing",
        "natural language",
        "computational linguistics"
    ]
    
    # Allow command line argument for custom concept
    if len(sys.argv) > 1:
        test_concept = sys.argv[1]
        test_variations = [test_concept]
        print(f"Using command line concept: {test_concept}")
    else:
        print(f"Using default test concept: {test_concept}")
        print(f"With variations: {test_variations}")
    
    try:
        result = query_post_training_local(test_concept, [])
        
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY:")
        print("="*60)
        print(f"Concept: {test_concept}")
        
        # Display results for all variations
        if result['all_variations_data']:
            print(f"Variations found: {len(result['all_variations_data'])}")
            
            total_doc_count = 0
            total_ngram_count = 0
            best_variation = None
            best_doc_count = 0
            
            for variation, data in result['all_variations_data'].items():
                doc_count = data.get('doc_count', 0)
                ngram_count = data.get('ngram_count', 0)
                total_doc_count += doc_count
                total_ngram_count += ngram_count
                
                print(f"\nVariation: {variation}")
                print(f"  Document count: {doc_count}")
                print(f"  N-gram count: {ngram_count}")
                print(f"  Tokens: {data.get('tokens', '')}")
                print(f"  Document contexts: {len(data.get('document_contexts', []))}")
                
                if doc_count > best_doc_count:
                    best_doc_count = doc_count
                    best_variation = variation
            
            print(f"\nTOTAL ACROSS ALL VARIATIONS:")
            print(f"Total document count: {total_doc_count}")
            print(f"Total n-gram count: {total_ngram_count}")
            print(f"Best variation: {best_variation} (with {best_doc_count} docs)")
            
            # Show sample contexts from the best variation
            if best_variation and best_variation in result['variations_contexts']:
                contexts = result['variations_contexts'][best_variation]
                if contexts:
                    print(f"\nSample document contexts from '{best_variation}':")
                    for i, ctx in enumerate(contexts[:3]):
                        print(f"\nDocument {i+1}:")
                        print(f"  Shard: {ctx.get('shard', 'N/A')}")
                        print(f"  Rank/Ptr: {ctx.get('rank', ctx.get('ptr', 'N/A'))}")
                        print(f"  Doc length: {ctx.get('doc_len', 'N/A')}")
                        print(f"  Text preview: {ctx['text'][:200]}...")
        else:
            print("No variations found with data")
            total_doc_count = 0
        
        print(f"\nPost-training frequency for '{test_concept}': {total_doc_count}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 