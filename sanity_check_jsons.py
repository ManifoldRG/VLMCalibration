import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

from tqdm import tqdm

from utils import (
    parse_gsm8k_answer,
    parse_mmlu_answer,
    parse_medmcqa_answer,
    parse_simpleqa_answer,
)


DATASET_PARSERS: Dict[str, Callable[[str], object]] = {
    "gsm8k": parse_gsm8k_answer,
    "mmlu": parse_mmlu_answer,
    "medmcqa": parse_medmcqa_answer,
    "simpleqa": parse_simpleqa_answer,
}


def check_file(file_path: Path, dataset: str) -> Tuple[int, int, List[int]]:
    """Return (ok_count, bad_count, bad_indices) for a records JSON file.

    A record is marked bad if the dataset-specific parser returns None or an
    invalid value (e.g., not in {A..D} for multiple-choice datasets).
    """
    parser = DATASET_PARSERS.get(dataset)
    if parser is None:
        raise ValueError(f"Unknown dataset '{dataset}' for {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    ok = 0
    bad = 0
    bad_indices: List[int] = []

    for rec in records:
        response = rec.get("response")
        idx = rec.get("idx")
        try:
            parsed = parser(response) if isinstance(response, str) else None
        except Exception:
            parsed = None
        valid = parsed is not None
        if dataset in {"mmlu", "medmcqa"} and isinstance(parsed, str):
            valid = parsed in {"A", "B", "C", "D"}
        if dataset == "gsm8k" and parsed is not None:
            valid = isinstance(parsed, int)
        if valid:
            ok += 1
        else:
            bad += 1
            if isinstance(idx, int):
                bad_indices.append(idx)

    return ok, bad, bad_indices


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity-check JSON result files by re-parsing model responses using "
            "dataset-specific parsers in utils.py. Flags records where parsing "
            "fails or yields invalid values."
        )
    )
    parser.add_argument("root", nargs="?", default=".", help="Project root to scan (default: .)")
    parser.add_argument("--show-all", action="store_true", help="Print all files, not only those with issues.")
    parser.add_argument("--out", type=str, default=None, help="Optional CSV output path for the per-file summary.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    root = Path(args.root)
    if not root.exists():
        print(f"Root not found: {root}", file=sys.stderr)
        sys.exit(1)

    # Gather files under {exp_type}/{dataset}/*.json excluding *details*.json
    file_tuples: List[Tuple[Path, Path, Path]] = []
    for exp_type in sorted([p for p in root.iterdir() if p.is_dir()]):
        for dataset_dir in sorted([p for p in exp_type.iterdir() if p.is_dir()]):
            dataset_name = dataset_dir.name
            if dataset_name not in DATASET_PARSERS:
                continue
            for jf in sorted(dataset_dir.glob("*.json")):
                if "details" in jf.name.lower():
                    continue
                file_tuples.append((exp_type, dataset_dir, jf))

    # Process with progress bar
    lines: List[str] = ["ExpType,Dataset,File,OK,BAD,BadIndices"]
    failed_files: List[str] = []
    total_bad = 0
    total_ok = 0

    for exp_type, dataset_dir, jf in tqdm(file_tuples, desc="Sanity checking"):
        try:
            ok, bad, bad_indices = check_file(jf, dataset_dir.name)
            total_ok += ok
            total_bad += bad
            if args.show_all or bad > 0:
                indices_str = ";".join(map(str, bad_indices)) if bad_indices else ""
                lines.append(f"{exp_type.name},{dataset_dir.name},{jf.name},{ok},{bad},{indices_str}")
        except Exception as exc:  # noqa: BLE00
            failed_files.append(f"{jf}: {exc}")

    report = "\n".join(lines)
    # print(report)

    # Print grand totals
    print(f"\nTotal OK records: {total_ok}")
    print(f"Total BAD records: {total_bad}")

    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"Saved summary to {args.out}")

    if failed_files:
        print("\nFiles that failed to validate:", file=sys.stderr)
        for msg in failed_files:
            print(f" - {msg}", file=sys.stderr)


if __name__ == "__main__":
    main()
