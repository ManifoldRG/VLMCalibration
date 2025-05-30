#!/usr/bin/env python3
"""
collect_infini_sync.py
Usage example
-------------
python collect_infini_sync.py \
    --zs zs.json --cot cot.json \
    --index v4_olmo-2-1124-13b-instruct_llama \
    --max-docs 3 \
    --out infinigram_hits.jsonl
"""
import argparse, json, pathlib, time, random
from typing import Dict, Any, List

import requests

API_URL = "https://api.infini-gram.io/"

# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def post(payload: Dict[str, Any], retries: int = 3, backoff: float = 0.5) -> Dict:
    """POST once with simple retry logic."""
    for attempt in range(retries):
        try:
            r = requests.post(API_URL, json=payload, timeout=30)
            r.raise_for_status()
            data = r.json()
            if "error" in data:
                raise RuntimeError(data["error"])
            return data
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(backoff * (2 ** attempt))
    return {}  # never reached

def count_ngram(index: str, text: str) -> int:
    payload = {"index": index, "query_type": "count", "query": text}
    return int(post(payload).get("count", 0))

def find_docs(index: str, text: str,
              max_docs: int = 3, window: int = 40) -> List[Dict[str, Any]]:
    """Return ≤ max_docs snippets that contain *text*."""
    payload_find = {"index": index, "query_type": "find", "query": text}
    meta = post(payload_find)
    segs = meta.get("segment_by_shard", [])
    hits = []
    for shard_id, (lo, hi) in enumerate(segs):
        if len(hits) >= max_docs or hi <= lo:
            continue
        rank = lo  # take first hit in this shard
        payload_doc = {
            "index": index,
            "query_type": "get_doc_by_rank",
            "s": shard_id,
            "rank": rank,
            "max_disp_len": window,
        }
        doc = post(payload_doc)
        hits.append({
            "shard": shard_id,
            "rank": rank,
            "snippet": " ".join(doc.get("tokens", [])),
            "doc_ix": doc.get("doc_ix"),
        })
    return hits

# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def process_pair(index: str, qa: Dict[str, Any],
                 is_cot: bool, max_docs: int) -> Dict[str, Any]:
    q = qa["question"].strip()
    ans = qa["response"].strip()
    correct = qa.get("correct")
    joint = f"{q} {ans}"

    ans_cnt = count_ngram(index, ans)
    joint_cnt = count_ngram(index, joint)

    record = {
        "idx": qa.get("idx"),
        "cot": is_cot,
        "correct": correct,
        "answer_count": ans_cnt,
        "joint_count": joint_cnt,
    }

    if not correct and max_docs:
        record["doc_snippets"] = find_docs(index, ans, max_docs=max_docs)
    return record

def main(args):
    zs_data = json.loads(pathlib.Path(args.zs).read_text())
    cot_data = json.loads(pathlib.Path(args.cot).read_text())

    with open(args.out, "w", encoding="utf8") as fh:
        for qa in zs_data:
            rec = process_pair(args.index, qa, False, args.max_docs)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        for qa in cot_data:
            rec = process_pair(args.index, qa, True, args.max_docs)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zs", required=True, help="zero-shot JSON file")
    ap.add_argument("--cot", required=True, help="chain-of-thought JSON file")
    ap.add_argument("--index", default="v4_olmo-2-1124-13b-instruct_llama",
                    help="Infini-gram index name")
    ap.add_argument("--max-docs", type=int, default=0,
                    help="# snippets for wrong answers (0 = skip)")
    ap.add_argument("--out", required=True, help="output JSONL path")
    args = ap.parse_args()
    main(args)
