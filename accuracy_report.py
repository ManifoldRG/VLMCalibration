import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from tqdm import tqdm


def infer_model_name(file_path: Path) -> str:
	"""Infer model name from filename by taking the substring after the last underscore.

	Falls back to the stem if no underscore exists.
	"""
	stem = file_path.stem
	if "_" in stem:
		return stem.split("_")[-1]
	return stem


def compute_accuracy_from_file(file_path: Path) -> Tuple[float, int]:
	"""Compute accuracy from a records JSON file containing a list of dicts with 'correct'.

	Returns a tuple (accuracy, count). Raises on JSON parsing errors.
	"""
	with file_path.open("r", encoding="utf-8") as f:
		records = json.load(f)
	if not isinstance(records, list):
		raise ValueError("Expected top-level list of records")
	valid = [bool(r.get("correct")) for r in records if isinstance(r, dict) and "correct" in r]
	if not valid:
		raise ValueError("No valid 'correct' entries found")
	acc = sum(1 for v in valid if v) / len(valid)
	return acc, len(valid)


def main(argv: Iterable[str] | None = None) -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Aggregate method-wise accuracies per model. Scans {exp_type}/{dataset} for JSON files, "
			"skipping names containing 'details'. Computes per-file accuracy from the 'correct' field, "
			"then macro-averages per dataset and reports per-method per-model macro accuracy."
		)
	)
	parser.add_argument("root", nargs="?", default=".", help="Project root to scan (default: .)")
	parser.add_argument("--out", type=str, default=None, help="Optional CSV output path")
	args = parser.parse_args(list(argv) if argv is not None else None)

	root = Path(args.root)
	if not root.exists():
		print(f"Root not found: {root}", file=sys.stderr)
		sys.exit(1)

	# Structure: {exp_type}/{dataset}/*.json (excluding *details*.json)
	per_model_method_dataset: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
	failed_files: List[str] = []

	# Collect all files first to get total count for progress bar
	all_files: List[Tuple[Path, Path, Path]] = []
	for exp_type in sorted([p for p in root.iterdir() if p.is_dir()]):
		# only consider directories that themselves have subdirectories with jsons
		for dataset_dir in sorted([p for p in exp_type.iterdir() if p.is_dir()]):
			json_files = sorted([p for p in dataset_dir.glob("*.json") if "details" not in p.name.lower()])
			for jf in json_files:
				all_files.append((exp_type, dataset_dir, jf))

	# Process files with progress bar
	for exp_type, dataset_dir, jf in tqdm(all_files, desc="Processing files"):
		try:
			acc, _ = compute_accuracy_from_file(jf)
			model = infer_model_name(jf)
			per_model_method_dataset[model][exp_type.name][dataset_dir.name] = acc
		except Exception as exc:  # noqa: BLE001
			failed_files.append(f"{jf}: {exc}")

	# Build CSV-like output: Model,Method,MacroAcc,NumDatasets,Datasets
	# Order: zs_exp, cot_exp, verbalized, verbalized_cot, otherAI
	method_order = ["zs_exp", "cot_exp", "verbalized", "verbalized_cot", "otherAI"]
	lines: List[str] = ["Model,Method,MacroAcc,NumDatasets,Datasets"]
	for model in sorted(per_model_method_dataset.keys()):
		methods = per_model_method_dataset[model]
		# First output methods in the specified order
		for method in method_order:
			if method in methods:
				ds_map = methods[method]
				if not ds_map:
					continue
				macro = sum(ds_map.values()) / len(ds_map)
				datasets_str = ";".join(f"{d}:{ds_map[d]:.4f}" for d in sorted(ds_map.keys()))
				lines.append(f"{model},{method},{macro:.6f},{len(ds_map)},{datasets_str}")
		# Then output any remaining methods not in the specified order
		for method in sorted(methods.keys()):
			if method not in method_order:
				ds_map = methods[method]
				if not ds_map:
					continue
				macro = sum(ds_map.values()) / len(ds_map)
				datasets_str = ";".join(f"{d}:{ds_map[d]:.4f}" for d in sorted(ds_map.keys()))
				lines.append(f"{model},{method},{macro:.6f},{len(ds_map)},{datasets_str}")

	output = "\n".join(lines)
	print(output)

	if args.out:
		Path(args.out).write_text(output + "\n", encoding="utf-8")
		print(f"Saved summary to {args.out}")

	# Flag failures at the end
	if failed_files:
		print("\nFiles that failed to parse:", file=sys.stderr)
		for msg in failed_files:
			print(f" - {msg}", file=sys.stderr)


if __name__ == "__main__":
	main()


