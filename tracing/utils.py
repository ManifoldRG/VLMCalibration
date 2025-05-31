import json
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from typing import List, Dict, Any, Tuple


def load_simpleqa_data(input_file: str, limit: int = None) -> List[Dict[str, Any]]:
    """Load SimpleQA data from JSON file."""
    print(f"Loading data from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if limit:
        data = data[:limit]
    
    print(f"Loaded {len(data)} records")
    return data


def plot_confidence_vs_frequency(raw_results: List[Dict], output_file: str):
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


def save_results(results: Dict[str, Any], output_file: str):
    """Save analysis results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}") 