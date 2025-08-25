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
    validate_gsm8k_answer,
    validate_mmlu_answer,
    validate_medmcqa_answer,
    validate_simpleqa_answer,
)


DATASET_PARSERS: Dict[str, Callable[[str], object]] = {
    "gsm8k": parse_gsm8k_answer,
    "mmlu": parse_mmlu_answer,
    "medmcqa": parse_medmcqa_answer,
    "simpleqa": parse_simpleqa_answer,
}

SKIP_REPARSE_DATASETS = {"gsm8k", "simpleqa", "truthfulqa"}


def _compute_correct(dataset: str, record: Dict, parsed_answer: object) -> object:
    """Return the correctness flag for a record using validators in utils.py.

    For datasets that need external graders (e.g., simpleqa), this function
    preserves the existing value if a client is not available.
    """
    true_answer = record.get("true_answer")

    if dataset == "gsm8k":
        try:
            return validate_gsm8k_answer(parsed_answer, true_answer)
        except Exception:
            return record.get("correct")
    if dataset == "mmlu":
        try:
            return validate_mmlu_answer(parsed_answer, true_answer)
        except Exception:
            return record.get("correct")
    if dataset == "medmcqa":
        try:
            return validate_medmcqa_answer(parsed_answer, true_answer)
        except Exception:
            return record.get("correct")
    if dataset == "simpleqa":
        # validate_simpleqa_answer requires an external client; preserve existing
        # value rather than failing when unavailable.
        return record.get("correct")

    return record.get("correct")


def reparse_and_write(src_file: Path, dataset: str, dst_file: Path) -> Tuple[int, int, int]:
    """Reparse a records JSON file and write updated records to ``dst_file``.

    Only the ``answer`` and ``correct`` fields are updated per record; all other
    fields are preserved as-is.

    Returns (num_total, num_answer_set, num_correct_set).
    """
    parser = DATASET_PARSERS.get(dataset)
    if parser is None:
        raise ValueError(f"Unknown dataset '{dataset}' for {src_file}")

    with src_file.open("r", encoding="utf-8") as f:
        records = json.load(f)

    total = 0
    n_set_answer = 0
    n_set_correct = 0
    new_records: List[Dict] = []

    for rec in records:
        total += 1
        response = rec.get("response")
        try:
            parsed_answer = parser(response) if isinstance(response, str) else None
        except Exception:
            parsed_answer = None

        # Update a shallow copy to avoid mutating original in-memory data
        out_rec = dict(rec)
        out_rec["answer"] = parsed_answer
        if parsed_answer is not None:
            n_set_answer += 1

        correct_value = _compute_correct(dataset, rec, parsed_answer)
        out_rec["correct"] = correct_value
        if correct_value is not None:
            n_set_correct += 1

        new_records.append(out_rec)

    dst_file.parent.mkdir(parents=True, exist_ok=True)
    with dst_file.open("w", encoding="utf-8") as f:
        json.dump(new_records, f, ensure_ascii=False, indent=2)

    return total, n_set_answer, n_set_correct


def copy_json_unchanged(src_file: Path, dst_file: Path) -> None:
    """Copy the JSON file byte-for-byte without modification."""
    dst_file.parent.mkdir(parents=True, exist_ok=True)
    dst_file.write_bytes(src_file.read_bytes())


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


def check_ptrue_file(file_path: Path) -> Tuple[int, int, List[int]]:
    """Return (ok_count, bad_count, bad_indices) for p_true range validity.

    Valid if p_true exists and is a number between 0 and 1 inclusive.
    """
    with file_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    ok = 0
    bad = 0
    bad_indices: List[int] = []

    for rec in records:
        idx = rec.get("idx")
        val = rec.get("p_true")
        # Treat None and booleans as invalid explicitly; JSON booleans are ints in Python.
        if val is None or isinstance(val, bool):
            valid = False
        else:
            try:
                v = float(val)
                valid = 0.0 <= v <= 1.0
            except (TypeError, ValueError):
                valid = False
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
    parser.add_argument("--ptrue-out", type=str, default=None, help="Optional CSV output path for p_true validity summary.")
    parser.add_argument(
        "--write-reparsed",
        action="store_true",
        help=(
            "Write reparsed JSONs under 'reparsed_results/{exp}/{dataset}/'. "
            "Only 'answer' and 'correct' fields are updated."
        ),
    )
    parser.add_argument(
        "--dst-root",
        type=str,
        default="reparsed_results",
        help="Destination root for reparsed outputs (default: reparsed_results)",
    )
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
            if dataset_name not in set(DATASET_PARSERS.keys()) | SKIP_REPARSE_DATASETS:
                continue
            for jf in sorted(dataset_dir.glob("*.json")):
                if "details" in jf.name.lower():
                    continue
                file_tuples.append((exp_type, dataset_dir, jf))

    # Process with progress bar
    lines: List[str] = ["ExpType,Dataset,File,OK,BAD,BadIndices"]
    ptrue_lines: List[str] = ["ExpType,Dataset,File,OK,BAD,BadIndices"]
    failed_files: List[str] = []
    total_bad = 0
    total_ok = 0
    total_written = 0
    total_answer_set = 0
    total_correct_set = 0

    last_shown_file: Path | None = None

    for exp_type, dataset_dir, jf in tqdm(file_tuples, desc="Sanity checking"):
        if last_shown_file != jf:
            print(f"\nChecking file: {exp_type.name}/{dataset_dir.name}/{jf.name}", file=sys.stderr)
            last_shown_file = jf
        try:
            if dataset_dir.name in DATASET_PARSERS:
                ok, bad, bad_indices = check_file(jf, dataset_dir.name)
                total_ok += ok
                total_bad += bad
                if args.show_all or bad > 0:
                    indices_str = ";".join(map(str, bad_indices)) if bad_indices else ""
                    lines.append(f"{exp_type.name},{dataset_dir.name},{jf.name},{ok},{bad},{indices_str}")
            else:
                if args.show_all:
                    lines.append(f"{exp_type.name},{dataset_dir.name},{jf.name},NA,NA,")

            if args.write_reparsed:
                dst_root = Path(args.dst_root)
                dst_path = dst_root / exp_type.name / dataset_dir.name / jf.name
                try:
                    if dataset_dir.name in SKIP_REPARSE_DATASETS:
                        copy_json_unchanged(jf, dst_path)
                        # Count totals for visibility only
                        total_written += 0
                    else:
                        tot, ans_set, cor_set = reparse_and_write(jf, dataset_dir.name, dst_path)
                        total_written += tot
                        total_answer_set += ans_set
                        total_correct_set += cor_set
                except Exception as exc2:  # noqa: BLE00
                    failed_files.append(f"{jf} (write): {exc2}")

            # p_true range check summary
            try:
                p_ok, p_bad, p_bad_indices = check_ptrue_file(jf)
                if args.show_all or p_bad > 0:
                    p_idx_str = ";".join(map(str, p_bad_indices)) if p_bad_indices else ""
                    ptrue_lines.append(f"{exp_type.name},{dataset_dir.name},{jf.name},{p_ok},{p_bad},{p_idx_str}")
            except Exception as exc3:  # noqa: BLE00
                failed_files.append(f"{jf} (p_true): {exc3}")
        except Exception as exc:  # noqa: BLE00
            failed_files.append(f"{jf}: {exc}")

    report = "\n".join(lines)
    ptrue_report = "\n".join(ptrue_lines)

    # Print grand totals
    print(f"\nTotal OK records: {total_ok}")
    print(f"Total BAD records: {total_bad}")

    if args.write_reparsed:
        print(
            f"Reparsed and wrote {total_written} records. "
            f"Answers set: {total_answer_set}, Correct set: {total_correct_set}"
        )

    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"Saved summary to {args.out}")
    if args.ptrue_out:
        Path(args.ptrue_out).write_text(ptrue_report + "\n", encoding="utf-8")
        print(f"Saved p_true summary to {args.ptrue_out}")

    if failed_files:
        print("\nFiles that failed to validate:", file=sys.stderr)
        for msg in failed_files:
            print(f" - {msg}", file=sys.stderr)


if __name__ == "__main__":
    main()
