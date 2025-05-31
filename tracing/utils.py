import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from typing import List, Dict, Any, Tuple


def load_simpleqa_data(input_file: str, limit: int = None, overconfident: bool = False, underconfident: bool = False) -> List[Dict[str, Any]]:
    """Load SimpleQA data from JSON file with optional filtering scenarios."""
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_count = len(data)
    
    # Apply filtering scenarios
    if overconfident:
        data = [record for record in data if record.get("p_true", 0) > 0.7 and record.get("correct", True) is False]
        print(f"Filtered for overconfident cases (p_true > 0.7 and correct = False): {len(data)} records")
    elif underconfident:
        data = [record for record in data if record.get("p_true", 1) < 0.4 and record.get("correct", False) is True]
        print(f"Filtered for underconfident cases (p_true < 0.4 and correct = True): {len(data)} records")
    
    if limit:
        data = data[:limit]
    
    print(f"Loaded {len(data)} records (from {original_count} total)")
    return data


def plot_confidence_vs_frequency(raw_results: List[Dict], output_file: str, overconfident: bool = False, underconfident: bool = False, cot_correct_zs_incorrect: bool = False, cot_incorrect_zs_correct: bool = False):
    """Plot p_true vs document frequency and p_true vs n-gram count (pretraining + post-training) as separate plots with both linear and log scales."""
    if not raw_results:
        print("No data available for plotting")
        return
    
    # Filter out entries where both doc frequency and n-gram count are zero
    filtered_results = []
    for r in raw_results:
        if r["combined_freq"] > 0 or r["combined_ngram_count"] > 0:
            filtered_results.append(r)
    
    if not filtered_results:
        print("No data with non-zero frequencies available for plotting")
        return
    
    # Extract data for plotting
    p_true_values = [r["p_true"] for r in filtered_results]
    combined_freqs = [r["combined_freq"] for r in filtered_results]
    combined_ngrams = [r["combined_ngram_count"] for r in filtered_results]
    concepts = [r["concept"] for r in filtered_results]
    
    # Create base filename without extension
    base_name = output_file.rsplit('.', 1)[0] if '.' in output_file else output_file
    extension = output_file.rsplit('.', 1)[1] if '.' in output_file else 'png'
    
    # Add filtering scenario to filename
    scenario_suffix = ""
    if overconfident:
        scenario_suffix = "_overconfident"
    elif underconfident:
        scenario_suffix = "_underconfident"
    elif cot_correct_zs_incorrect:
        scenario_suffix = "_cot_correct_zs_incorrect"
    elif cot_incorrect_zs_correct:
        scenario_suffix = "_cot_incorrect_zs_correct"
    
    # Update title based on filtering scenario
    title_prefix = ""
    if overconfident:
        title_prefix = "Overconfident Cases: "
    elif underconfident:
        title_prefix = "Underconfident Cases: "
    elif cot_correct_zs_incorrect:
        title_prefix = "CoT Correct, ZS Incorrect: "
    elif cot_incorrect_zs_correct:
        title_prefix = "CoT Incorrect, ZS Correct: "
    
    # Check if we should create log scale plots
    freq_needs_log = max(combined_freqs) > 0 and max(combined_freqs) > 10 * min([f for f in combined_freqs if f > 0], default=1)
    ngram_needs_log = max(combined_ngrams) > 0 and max(combined_ngrams) > 10 * min([n for n in combined_ngrams if n > 0], default=1)
    
    # Helper function to add annotations
    def add_annotations(ax, x_data, y_data, concepts, threshold_factor=0.8):
        for i, (x, y, concept) in enumerate(zip(x_data, y_data, concepts)):
            if x > max(x_data) * threshold_factor or y > 0.95 or y < 0.1:
                ax.annotate(concept[:20] + "..." if len(concept) > 20 else concept, 
                           (x, y), xytext=(5, 5), textcoords='offset points',
                           fontsize=7, alpha=0.7)
    
    # Helper function to add trend line
    def add_trend_line(ax, x_data, y_data, label_prefix="Trend"):
        if len(set(x_data)) > 1:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(x_data), max(x_data), 100)
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2,
                   label=f'{label_prefix}: y={z[0]:.4f}x+{z[1]:.3f}')
            ax.legend()
    
    # Plot 1: Document Frequency - Linear Scale
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1.scatter(combined_freqs, p_true_values, alpha=0.7, color='blue', s=60)
    ax1.set_xlabel('Combined Document Frequency (Pretraining + Post-training)')
    ax1.set_ylabel('p_true (Model Confidence)')
    ax1.set_title(f'{title_prefix}Model Confidence vs Document Frequency (Linear Scale)')
    ax1.grid(True, alpha=0.3)
    add_trend_line(ax1, combined_freqs, p_true_values)
    add_annotations(ax1, combined_freqs, p_true_values, concepts)
    
    doc_freq_linear_file = f"{base_name}{scenario_suffix}_doc_freq_linear.{extension}"
    plt.tight_layout()
    plt.savefig(doc_freq_linear_file, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print(f"Document frequency plot (linear) saved to {doc_freq_linear_file}")
    
    # Plot 2: Document Frequency - Log Scale (if needed)
    if freq_needs_log:
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        ax2.scatter(combined_freqs, p_true_values, alpha=0.7, color='blue', s=60)
        ax2.set_xlabel('Combined Document Frequency (Pretraining + Post-training)')
        ax2.set_ylabel('p_true (Model Confidence)')
        ax2.set_title(f'{title_prefix}Model Confidence vs Document Frequency (Log Scale)')
        ax2.grid(True, alpha=0.3)
        ax2.set_xscale('log')
        add_trend_line(ax2, combined_freqs, p_true_values, "Linear Trend")
        add_annotations(ax2, combined_freqs, p_true_values, concepts)
        
        doc_freq_log_file = f"{base_name}{scenario_suffix}_doc_freq_log.{extension}"
        plt.tight_layout()
        plt.savefig(doc_freq_log_file, dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"Document frequency plot (log) saved to {doc_freq_log_file}")
    
    # Plot 3: N-gram Count - Linear Scale
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    ax3.scatter(combined_ngrams, p_true_values, alpha=0.7, color='green', s=60)
    ax3.set_xlabel('Combined N-gram Count (Pretraining + Post-training)')
    ax3.set_ylabel('p_true (Model Confidence)')
    ax3.set_title(f'{title_prefix}Model Confidence vs N-gram Count (Linear Scale)')
    ax3.grid(True, alpha=0.3)
    add_trend_line(ax3, combined_ngrams, p_true_values)
    add_annotations(ax3, combined_ngrams, p_true_values, concepts)
    
    ngram_linear_file = f"{base_name}{scenario_suffix}_ngram_count_linear.{extension}"
    plt.tight_layout()
    plt.savefig(ngram_linear_file, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print(f"N-gram count plot (linear) saved to {ngram_linear_file}")
    
    # Plot 4: N-gram Count - Log Scale (if needed)
    if ngram_needs_log:
        fig4, ax4 = plt.subplots(figsize=(12, 8))
        ax4.scatter(combined_ngrams, p_true_values, alpha=0.7, color='green', s=60)
        ax4.set_xlabel('Combined N-gram Count (Pretraining + Post-training)')
        ax4.set_ylabel('p_true (Model Confidence)')
        ax4.set_title(f'{title_prefix}Model Confidence vs N-gram Count (Log Scale)')
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')
        add_trend_line(ax4, combined_ngrams, p_true_values, "Linear Trend")
        add_annotations(ax4, combined_ngrams, p_true_values, concepts)
        
        ngram_log_file = f"{base_name}{scenario_suffix}_ngram_count_log.{extension}"
        plt.tight_layout()
        plt.savefig(ngram_log_file, dpi=300, bbox_inches='tight')
        plt.close(fig4)
        print(f"N-gram count plot (log) saved to {ngram_log_file}")
    
    # Print summary statistics
    scenario_text = ""
    if overconfident:
        scenario_text = " (Overconfident Cases)"
    elif underconfident:
        scenario_text = " (Underconfident Cases)"
    elif cot_correct_zs_incorrect:
        scenario_text = " (CoT Correct, ZS Incorrect)"
    elif cot_incorrect_zs_correct:
        scenario_text = " (CoT Incorrect, ZS Correct)"
    
    print(f"\nPlot Summary{scenario_text}:")
    print(f"Total questions (after filtering): {len(filtered_results)}")
    print(f"Questions filtered out (zero frequencies): {len(raw_results) - len(filtered_results)}")
    print(f"p_true range: [{min(p_true_values):.3f}, {max(p_true_values):.3f}]")
    print(f"Document frequency range: [{min(combined_freqs)}, {max(combined_freqs)}]")
    print(f"N-gram count range: [{min(combined_ngrams)}, {max(combined_ngrams)}]")
    print(f"Created log scale plots: doc_freq={freq_needs_log}, ngram_count={ngram_needs_log}")
    
    # Calculate correlations
    if len(set(combined_freqs)) > 1:
        corr_freq, p_val_freq = stats.pearsonr(combined_freqs, p_true_values)
        print(f"Document frequency correlation: r={corr_freq:.4f}, p={p_val_freq:.4f}")
    
    if len(set(combined_ngrams)) > 1:
        corr_ngram, p_val_ngram = stats.pearsonr(combined_ngrams, p_true_values)
        print(f"N-gram count correlation: r={corr_ngram:.4f}, p={p_val_ngram:.4f}")


def save_results(results: Dict[str, Any], output_file: str):
    """Save analysis results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


def load_and_compare_zs_cot_data(zs_file: str, cot_file: str, limit: int = None, cot_correct_zs_incorrect: bool = False, cot_incorrect_zs_correct: bool = False) -> List[Dict[str, Any]]:
    """Load and compare data from zero-shot and chain-of-thought JSON files with filtering scenarios."""
    print(f"Loading zero-shot data from {zs_file}...")
    with open(zs_file, 'r', encoding='utf-8') as f:
        zs_data = json.load(f)
    
    print(f"Loading chain-of-thought data from {cot_file}...")
    with open(cot_file, 'r', encoding='utf-8') as f:
        cot_data = json.load(f)
    
    # Create lookup dictionary for CoT data using idx
    cot_lookup = {record.get("idx", i): record for i, record in enumerate(cot_data)}
    
    # Find matching records and apply comparison filtering
    compared_data = []
    for i, zs_record in enumerate(zs_data):
        zs_idx = zs_record.get("idx", i)
        
        if zs_idx in cot_lookup:
            cot_record = cot_lookup[zs_idx]
            
            zs_correct = zs_record.get("correct", False)
            cot_correct = cot_record.get("correct", False)
            
            # Apply filtering scenarios
            if cot_correct_zs_incorrect:
                if cot_correct and not zs_correct:
                    # Use CoT record but keep both results for reference
                    combined_record = cot_record.copy()
                    combined_record["zs_correct"] = zs_correct
                    combined_record["cot_correct"] = cot_correct
                    combined_record["zs_p_true"] = zs_record.get("p_true", 0)
                    compared_data.append(combined_record)
            elif cot_incorrect_zs_correct:
                if not cot_correct and zs_correct:
                    # Use ZS record but keep both results for reference
                    combined_record = zs_record.copy()
                    combined_record["zs_correct"] = zs_correct
                    combined_record["cot_correct"] = cot_correct
                    combined_record["cot_p_true"] = cot_record.get("p_true", 0)
                    compared_data.append(combined_record)
    
    original_count = len(zs_data)
    
    if cot_correct_zs_incorrect:
        print(f"Filtered for cases where CoT is correct but ZS is incorrect: {len(compared_data)} records")
    elif cot_incorrect_zs_correct:
        print(f"Filtered for cases where CoT is incorrect but ZS is correct: {len(compared_data)} records")
    
    if limit:
        compared_data = compared_data[:limit]
    
    print(f"Loaded {len(compared_data)} records (from {original_count} total ZS records)")
    return compared_data 