import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_confidence_and_correct(json_path: str) -> Tuple[np.ndarray, np.ndarray]:

	"""Load `p_true` confidences and correctness flags from a records JSON file.

	Parameters
	----------
	json_path: str
		Path to the JSON file that contains a list of records, each with
		`p_true` and `correct` fields.

	Returns
	-------
	Tuple[np.ndarray, np.ndarray]
		A tuple of (confidences, correct_flags) as numpy arrays.
	"""

	p = Path(json_path)
	with p.open("r", encoding="utf-8") as f:
		data = json.load(f)

	confidences: list[float] = []
	correct_flags: list[bool] = []
	for record in data:
		p_true = record.get("p_true")
		correct = record.get("correct")
		if p_true is None or correct is None:
			continue
		try:
			p_val = float(p_true)
		except (TypeError, ValueError):
			continue
		if math.isnan(p_val):
			continue
		# Clamp to [0, 1] just in case
		p_val = max(0.0, min(1.0, p_val))
		confidences.append(p_val)
		correct_flags.append(bool(correct))

	return np.asarray(confidences, dtype=np.float64), np.asarray(correct_flags, dtype=bool)


def plot_confidence_histograms(
	confidences: np.ndarray,
	correct_flags: np.ndarray,
	output_basepath: str,
	bins: int = 30,
) -> None:

	"""Plot overlaid histograms of confidence for correct vs incorrect.

	This function saves two PNGs:
	- "..._conf_hist_raw.png" (density=False)
	- "..._conf_hist_density.png" (density=True)

	Parameters
	----------
	confidences: np.ndarray
		Array of model confidence values in [0, 1].
	correct_flags: np.ndarray
		Boolean array indicating correctness of the prediction.
	output_basepath: str
		Base path (without extension) for output images.
	bins: int
		Number of histogram bins.
	"""

	mask_correct = correct_flags
	mask_incorrect = ~correct_flags
	conf_correct = confidences[mask_correct]
	conf_incorrect = confidences[mask_incorrect]
	bin_edges = np.linspace(0.0, 1.0, bins + 1)

	for density in (False, True):
		fig, ax = plt.subplots(1, 1, figsize=(6, 4))
		ax.hist(
			conf_correct,
			bins=bin_edges,
			color="tab:green",
			alpha=0.6,
			edgecolor="white",
			density=density,
			label="Correct",
		)
		ax.hist(
			conf_incorrect,
			bins=bin_edges,
			color="tab:red",
			alpha=0.6,
			edgecolor="white",
			density=density,
			label="Incorrect",
		)

		if conf_correct.size > 0:
			mu_c = float(conf_correct.mean())
			ax.axvline(mu_c, color="tab:green", linestyle="--", linewidth=1)
			ax.text(
				mu_c,
				ax.get_ylim()[1] * (0.92 if density else 0.92),
				f"μ={mu_c:.3f}",
				fontsize=8,
				ha="center",
				va="top",
				color="tab:green",
			)
		if conf_incorrect.size > 0:
			mu_i = float(conf_incorrect.mean())
			ax.axvline(mu_i, color="tab:red", linestyle="--", linewidth=1)
			ax.text(
				mu_i,
				ax.get_ylim()[1] * (0.92 if density else 0.92),
				f"μ={mu_i:.3f}",
				fontsize=8,
				ha="center",
				va="top",
				color="tab:red",
			)

		ax.set_xlim(0.0, 1.0)
		ax.set_xlabel("Confidence (p_true)")
		ax.set_ylabel("Density" if density else "Count")
		ax.grid(True, linestyle=":", alpha=0.4)
		ax.legend(loc="upper left", fontsize=9)

		ax.set_title("Model Confidence Distributions", fontsize=12)
		fig.tight_layout()

		suffix = "density" if density else "raw"
		out_path = f"{output_basepath}_conf_hist_{suffix}.png"
		fig.savefig(out_path, dpi=200)
		plt.close(fig)


def plot_reliability_diagram(
	confidences: np.ndarray,
	correct_flags: np.ndarray,
	output_path: str,
	num_bins: int = 10,
	adaptive: bool = False,
	plot: bool = True,
) -> float:

	"""Plot reliability diagram and return the Expected Calibration Error (ECE).

	Also prints the Brier score and its Murphy (1973) decomposition into
	Reliability, Resolution, and Uncertainty, computed using the same binning
	as the diagram.

	Parameters
	----------
	confidences: np.ndarray
		Array of model confidence values in [0, 1].
	correct_flags: np.ndarray
		Boolean array indicating correctness of the prediction.
	output_path: str
		Output image path (.png).
	num_bins: int
		Number of bins to compute calibration.
	adaptive: bool
		If True, use equal-count (quantile) bins instead of equal-width bins.

	Returns
	-------
	float
		The ECE value computed with equal-width bins.
	"""

	# Choose bin edges: equal-width (default) or adaptive equal-count (quantiles)
	if adaptive:
		qs = np.linspace(0.0, 1.0, num_bins + 1)
		try:
			edges = np.quantile(confidences, qs, method="linear")
		except TypeError:
			# NumPy < 1.22
			edges = np.quantile(confidences, qs, interpolation="linear")
		edges = np.clip(edges, 0.0, 1.0)
		bin_edges = np.unique(edges)
		if bin_edges.size < 2:
			bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
	else:
		bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

	effective_bins = len(bin_edges) - 1
	# Map each confidence to a bin index in [0, num_bins-1]
	bin_indices = np.digitize(confidences, bin_edges[1:-1], right=False)

	bin_acc: list[float] = []
	bin_conf: list[float] = []
	bin_count: list[int] = []
	for b in range(effective_bins):
		mask = bin_indices == b
		count_b = int(mask.sum())
		if count_b == 0:
			bin_acc.append(np.nan)
			bin_conf.append((bin_edges[b] + bin_edges[b + 1]) / 2.0)
			bin_count.append(0)
			continue
		acc_b = float(correct_flags[mask].mean())
		conf_b = float(confidences[mask].mean())
		bin_acc.append(acc_b)
		bin_conf.append(conf_b)
		bin_count.append(count_b)

	# Expected Calibration Error (ECE)
	total = float(len(confidences))
	ece = 0.0
	for acc_b, conf_b, n_b in zip(bin_acc, bin_conf, bin_count):
		if n_b == 0 or math.isnan(acc_b):
			continue
		ece += (n_b / total) * abs(acc_b - conf_b)

	# Brier score and Murphy decomposition (using same bins)
	y_true = correct_flags.astype(float)
	brier = float(np.mean((confidences - y_true) ** 2))
	# Base rate
	base_rate = float(y_true.mean()) if total > 0 else float("nan")
	# Reliability, Resolution, Uncertainty
	reliability = 0.0
	resolution = 0.0
	uncertainty = base_rate * (1.0 - base_rate) if not math.isnan(base_rate) else float("nan")
	for acc_b, conf_b, n_b in zip(bin_acc, bin_conf, bin_count):
		if n_b == 0 or math.isnan(acc_b):
			continue
		w = n_b / total
		reliability += w * (conf_b - acc_b) ** 2
		resolution += w * (acc_b - base_rate) ** 2

	# Plot (optional)
	if plot:
		fig, ax = plt.subplots(figsize=(5, 5))
		ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")

		# Bar-style reliability diagram (accuracy per bin)
		non_empty = [i for i, n in enumerate(bin_count) if n > 0 and not math.isnan(bin_acc[i])]
		left_edges = bin_edges[:-1]
		bin_widths = np.diff(bin_edges)
		bar_lefts = [left_edges[i] for i in non_empty]
		bar_widths = [bin_widths[i] for i in non_empty]
		bar_heights = [bin_acc[i] for i in non_empty]
		ax.bar(
			bar_lefts,
			bar_heights,
			width=bar_widths,
			align="edge",
			color="tab:blue",
			alpha=0.7,
			edgecolor="white",
			label="Model (bin acc)",
		)

		ax.set_xlabel("Confidence")
		ax.set_ylabel("Accuracy")
		ax.set_xlim(0.0, 1.0)
		ax.set_ylim(0.0, 1.0)
		adapt_tag = ", adaptive" if adaptive else ""
		ax.set_title(
			f"Reliability Diagram (ECE={ece:.3f}, bins={effective_bins}{adapt_tag})\n"
			f"Brier={brier:.3f} | Rel={reliability:.3f}, Res={resolution:.3f}, Unc={uncertainty:.3f}"
		)
		ax.grid(True, linestyle=":", alpha=0.4)
		ax.legend(loc="lower right", fontsize=8)
		fig.tight_layout()
		fig.savefig(output_path, dpi=200)
		plt.close(fig)

	# Print Brier score and decomposition to stdout for quick inspection
	print(
		f"ECE: {ece:.6f} | Brier: {brier:.6f} | "
		f"Reliability: {reliability:.6f} | Resolution: {resolution:.6f} | Uncertainty: {uncertainty:.6f}"
	)

	return float(ece)


def _derive_output_base(json_path: str, prefix: str | None) -> str:
	p = Path(json_path)
	name = prefix if prefix else p.stem
	return str(p.with_name(name))


def main(argv: Iterable[str] | None = None) -> None:
	parser = argparse.ArgumentParser(
		description=(
			"Plot confidence histograms (raw and density) and a reliability diagram "
			"from zs_exp JSON records containing 'p_true' and 'correct'."
		)
	)
	parser.add_argument("json_path", help="Path to the records JSON file.")
	parser.add_argument(
		"--bins",
		type=int,
		default=30,
		help="Number of bins for confidence histograms.",
	)
	parser.add_argument(
		"--reliability-bins",
		type=int,
		default=10,
		help="Number of bins for the reliability diagram.",
	)
	parser.add_argument(
		"--adaptive",
		action="store_true",
		help="Use adaptive (equal-count) binning for the reliability diagram.",
	)
	parser.add_argument(
		"--no-plots",
		action="store_true",
		help="Disable plotting; only compute and print metrics.",
	)
	parser.add_argument(
		"--prefix",
		type=str,
		default=None,
		help=(
			"Optional output filename prefix (defaults to the input file's stem). "
			"Images are saved next to the input file."
		),
	)

	args = parser.parse_args(list(argv) if argv is not None else None)

	confidences, correct_flags = load_confidence_and_correct(args.json_path)
	output_base = _derive_output_base(args.json_path, args.prefix)

	# Accuracy
	accuracy = float(correct_flags.mean()) if len(correct_flags) > 0 else float("nan")
	print(f"Accuracy: {accuracy:.6f}")

	if not args.no_plots:
		plot_confidence_histograms(
			confidences=confidences,
			correct_flags=correct_flags,
			output_basepath=output_base,
			bins=args.bins,
		)

	reliability_path = f"{output_base}_reliability.png"
	plot_reliability_diagram(
		confidences=confidences,
		correct_flags=correct_flags,
		output_path=reliability_path,
		num_bins=args.reliability_bins,
		adaptive=args.adaptive,
		plot=not args.no_plots,
	)


if __name__ == "__main__":
	main()


