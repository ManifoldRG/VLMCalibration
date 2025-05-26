import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
import os
import json
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
warnings.filterwarnings('ignore')

def setup_publication_style():
    """Configure matplotlib for publication-quality figures"""
    import matplotlib as mpl
    
    # Font settings for publication
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    
    # Font sizes
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 10
    
    # Figure settings
    mpl.rcParams['figure.figsize'] = (10, 6)
    mpl.rcParams['figure.dpi'] = 300
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.1
    
    # Line and marker settings
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['lines.markersize'] = 6
    
    # Grid and axes
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.linewidth'] = 0.8

def discover_data_files(base_dirs: List[str]) -> Dict[str, List[str]]:
    """Discover all JSON data files excluding experiment_details"""
    data_files = {}
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
            
        # Find all JSON files that match the pattern
        pattern = os.path.join(base_dir, "*", "*_records_*.json")
        files = glob.glob(pattern)
        
        # Filter out experiment_details files
        files = [f for f in files if 'experiment_details' not in os.path.basename(f)]
        
        if files:
            data_files[base_dir] = files
    
    return data_files

def parse_filename(filepath: str) -> Dict[str, str]:
    """Parse filename to extract metadata"""
    filename = os.path.basename(filepath)
    parts = filename.replace('.json', '').split('_')
    
    # Extract components
    exp_type = f"{parts[0]}_{parts[1]}"  # cot_exp or zs_exp
    split = parts[3]     # train, val, test
    
    # Model name is everything after the fourth underscore
    model_name = '_'.join(parts[4:])
    
    # Dataset is the parent directory name
    dataset = os.path.basename(os.path.dirname(filepath))
    
    return {
        'exp_type': exp_type,
        'dataset': dataset,
        'split': split,
        'model_name': model_name,
        'filepath': filepath
    }

def load_json_data(filepath: str) -> Optional[pd.DataFrame]:
    """Load and process JSON data file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Check if 'correct' column exists
        if 'correct' not in df.columns:
            print(f"Warning: 'correct' column not found in {filepath}")
            return None
            
        # Check if 'p_true' column exists
        if 'p_true' not in df.columns:
            print(f"Warning: 'p_true' column not found in {filepath}")
            return None
            
        return df
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def calculate_ece(confidences: np.ndarray, correctness: np.ndarray, n_bins: int = 10) -> float:
    """
    Calculate Expected Calibration Error (ECE)
    
    Args:
        confidences: Array of confidence scores (0-1)
        correctness: Array of binary correctness indicators (0 or 1)
        n_bins: Number of bins for calibration
        
    Returns:
        ECE score
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    total_samples = len(confidences)
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = correctness[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            
            # Add weighted contribution to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def calculate_brier_score(confidences: np.ndarray, correctness: np.ndarray) -> float:
    """
    Calculate Brier Score
    
    Args:
        confidences: Array of confidence scores (0-1)
        correctness: Array of binary correctness indicators (0 or 1)
        
    Returns:
        Brier score
    """
    return np.mean((confidences - correctness) ** 2)

def calculate_metrics_for_file(filepath: str) -> Tuple[str, Optional[Dict], bool]:
    """Calculate calibration metrics for a single file"""
    try:
        metadata = parse_filename(filepath)
        df = load_json_data(filepath)
        
        if df is None:
            return filepath, None, False
        
        # Remove rows with None values in critical columns
        initial_len = len(df)
        df = df.dropna(subset=['correct', 'p_true'])
        
        if len(df) == 0:
            print(f"Warning: No valid data after removing None values in {filepath}")
            return filepath, None, False
        
        if len(df) != initial_len:
            print(f"Warning: Removed {initial_len - len(df)} rows with None values from {filepath}")
        
        # Convert correct column to numeric (True/False -> 1/0)
        correctness = df['correct'].astype(int).values
        confidences = df['p_true'].values
        
        # Calculate metrics
        ece = calculate_ece(confidences, correctness)
        brier = calculate_brier_score(confidences, correctness)
        
        # Additional metrics
        accuracy = correctness.mean()
        mean_confidence = confidences.mean()
        n_samples = len(df)
        
        result = {
            'exp_type': metadata['exp_type'],
            'dataset': metadata['dataset'],
            'split': metadata['split'],
            'model_name': metadata['model_name'],
            'ece': ece,
            'brier_score': brier,
            'accuracy': accuracy,
            'mean_confidence': mean_confidence,
            'n_samples': n_samples,
            'filepath': filepath
        }
        
        return filepath, result, True
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return filepath, None, False

def aggregate_model_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics per model across datasets"""
    
    # Group by experiment type and model
    grouped = metrics_df.groupby(['exp_type', 'model_name'])
    
    aggregated_results = []
    
    for (exp_type, model_name), group in grouped:
        # Calculate weighted averages (weighted by number of samples)
        total_samples = group['n_samples'].sum()
        
        # Weighted average for ECE and Brier score
        ece_weighted = (group['ece'] * group['n_samples']).sum() / total_samples
        brier_weighted = (group['brier_score'] * group['n_samples']).sum() / total_samples
        accuracy_weighted = (group['accuracy'] * group['n_samples']).sum() / total_samples
        confidence_weighted = (group['mean_confidence'] * group['n_samples']).sum() / total_samples
        
        # Simple averages (unweighted across datasets)
        ece_avg = group['ece'].mean()
        brier_avg = group['brier_score'].mean()
        accuracy_avg = group['accuracy'].mean()
        confidence_avg = group['mean_confidence'].mean()
        
        # Standard deviations across datasets
        ece_std = group['ece'].std()
        brier_std = group['brier_score'].std()
        accuracy_std = group['accuracy'].std()
        confidence_std = group['mean_confidence'].std()
        
        aggregated_results.append({
            'exp_type': exp_type,
            'model_name': model_name,
            'ece_weighted': ece_weighted,
            'ece_avg': ece_avg,
            'ece_std': ece_std,
            'brier_weighted': brier_weighted,
            'brier_avg': brier_avg,
            'brier_std': brier_std,
            'accuracy_weighted': accuracy_weighted,
            'accuracy_avg': accuracy_avg,
            'accuracy_std': accuracy_std,
            'confidence_weighted': confidence_weighted,
            'confidence_avg': confidence_avg,
            'confidence_std': confidence_std,
            'total_samples': total_samples,
            'n_datasets': len(group),
            'datasets': ', '.join(sorted(group['dataset'].unique()))
        })
    
    return pd.DataFrame(aggregated_results)

def create_calibration_comparison_plots(aggregated_df: pd.DataFrame, save_dir: str):
    """Create comparison plots for calibration metrics"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Separate CoT and ZS data
    cot_data = aggregated_df[aggregated_df['exp_type'] == 'cot_exp'].copy()
    zs_data = aggregated_df[aggregated_df['exp_type'] == 'zs_exp'].copy()
    
    if len(cot_data) == 0 or len(zs_data) == 0:
        print("Warning: Missing data for one experiment type - cannot create comparison plots")
        return
    
    # Plot 1: ECE Comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ECE comparison
    models = sorted(set(cot_data['model_name'].unique()) & set(zs_data['model_name'].unique()))
    
    cot_ece = [cot_data[cot_data['model_name'] == m]['ece_avg'].iloc[0] for m in models]
    zs_ece = [zs_data[zs_data['model_name'] == m]['ece_avg'].iloc[0] for m in models]
    
    # Create scatter plot
    colors = ['#2E8B57' if cot < zs else '#DC143C' for cot, zs in zip(cot_ece, zs_ece)]
    ax1.scatter(zs_ece, cot_ece, c=colors, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add diagonal line
    min_val = min(min(cot_ece), min(zs_ece))
    max_val = max(max(cot_ece), max(zs_ece))
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Equal ECE')
    
    # Add model labels
    for i, model in enumerate(models):
        model_short = model.replace('-', ' ').replace('_', ' ')
        if len(model_short) > 15:
            model_short = model_short[:12] + '...'
        ax1.annotate(model_short, (zs_ece[i], cot_ece[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
    
    ax1.set_xlabel('Zero-Shot ECE')
    ax1.set_ylabel('Chain-of-Thought ECE')
    ax1.set_title('Expected Calibration Error\nCoT vs Zero-Shot')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Brier Score comparison
    cot_brier = [cot_data[cot_data['model_name'] == m]['brier_avg'].iloc[0] for m in models]
    zs_brier = [zs_data[zs_data['model_name'] == m]['brier_avg'].iloc[0] for m in models]
    
    colors2 = ['#2E8B57' if cot < zs else '#DC143C' for cot, zs in zip(cot_brier, zs_brier)]
    ax2.scatter(zs_brier, cot_brier, c=colors2, s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Add diagonal line
    min_val2 = min(min(cot_brier), min(zs_brier))
    max_val2 = max(max(cot_brier), max(zs_brier))
    ax2.plot([min_val2, max_val2], [min_val2, max_val2], 'k--', alpha=0.5, label='Equal Brier')
    
    # Add model labels
    for i, model in enumerate(models):
        model_short = model.replace('-', ' ').replace('_', ' ')
        if len(model_short) > 15:
            model_short = model_short[:12] + '...'
        ax2.annotate(model_short, (zs_brier[i], cot_brier[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)
    
    ax2.set_xlabel('Zero-Shot Brier Score')
    ax2.set_ylabel('Chain-of-Thought Brier Score')
    ax2.set_title('Brier Score\nCoT vs Zero-Shot')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'calibration_metrics_comparison.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, 'calibration_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Plot 2: Model ranking plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # CoT ECE ranking
    cot_sorted = cot_data.sort_values('ece_avg')
    y_pos = np.arange(len(cot_sorted))
    bars1 = ax1.barh(y_pos, cot_sorted['ece_avg'], color='#2E8B57', alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([m.replace('-', ' ').replace('_', ' ') for m in cot_sorted['model_name']], fontsize=9)
    ax1.set_xlabel('ECE')
    ax1.set_title('Chain-of-Thought: ECE Ranking\n(Lower is Better)')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars1, cot_sorted['ece_avg'])):
        ax1.text(value + 0.001, i, f'{value:.3f}', va='center', fontsize=8)
    
    # ZS ECE ranking
    zs_sorted = zs_data.sort_values('ece_avg')
    y_pos = np.arange(len(zs_sorted))
    bars2 = ax2.barh(y_pos, zs_sorted['ece_avg'], color='#DC143C', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([m.replace('-', ' ').replace('_', ' ') for m in zs_sorted['model_name']], fontsize=9)
    ax2.set_xlabel('ECE')
    ax2.set_title('Zero-Shot: ECE Ranking\n(Lower is Better)')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars2, zs_sorted['ece_avg'])):
        ax2.text(value + 0.001, i, f'{value:.3f}', va='center', fontsize=8)
    
    # CoT Brier ranking
    cot_brier_sorted = cot_data.sort_values('brier_avg')
    y_pos = np.arange(len(cot_brier_sorted))
    bars3 = ax3.barh(y_pos, cot_brier_sorted['brier_avg'], color='#2E8B57', alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([m.replace('-', ' ').replace('_', ' ') for m in cot_brier_sorted['model_name']], fontsize=9)
    ax3.set_xlabel('Brier Score')
    ax3.set_title('Chain-of-Thought: Brier Score Ranking\n(Lower is Better)')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars3, cot_brier_sorted['brier_avg'])):
        ax3.text(value + 0.001, i, f'{value:.3f}', va='center', fontsize=8)
    
    # ZS Brier ranking
    zs_brier_sorted = zs_data.sort_values('brier_avg')
    y_pos = np.arange(len(zs_brier_sorted))
    bars4 = ax4.barh(y_pos, zs_brier_sorted['brier_avg'], color='#DC143C', alpha=0.7)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels([m.replace('-', ' ').replace('_', ' ') for m in zs_brier_sorted['model_name']], fontsize=9)
    ax4.set_xlabel('Brier Score')
    ax4.set_title('Zero-Shot: Brier Score Ranking\n(Lower is Better)')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars4, zs_brier_sorted['brier_avg'])):
        ax4.text(value + 0.001, i, f'{value:.3f}', va='center', fontsize=8)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'calibration_metrics_rankings.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, 'calibration_metrics_rankings.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    """Main function to calculate calibration metrics"""
    
    setup_publication_style()
    
    # Discover all data files
    base_dirs = ['cot_exp', 'zs_exp']
    data_files = discover_data_files(base_dirs)
    
    if not data_files:
        print("❌ No data files found!")
        return
    
    # Collect all file paths
    all_files = []
    for base_dir, files in data_files.items():
        all_files.extend(files)
    
    print(f"📊 Found {len(all_files)} data files across {len(data_files)} experiment types")
    
    # Process all files with ThreadPoolExecutor
    all_metrics = []
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(calculate_metrics_for_file, filepath): filepath 
                         for filepath in all_files}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(all_files), desc="Calculating calibration metrics", unit="file") as pbar:
            for future in as_completed(future_to_file):
                filepath = future_to_file[future]
                try:
                    result_filepath, metrics, success = future.result()
                    
                    if success and metrics is not None:
                        all_metrics.append(metrics)
                        pbar.set_postfix_str(f"✅ {os.path.basename(filepath)}")
                    else:
                        failed_files.append(result_filepath)
                        pbar.set_postfix_str(f"❌ {os.path.basename(filepath)}")
                        
                except Exception as e:
                    failed_files.append(filepath)
                    pbar.set_postfix_str(f"❌ {os.path.basename(filepath)} - {str(e)}")
                
                pbar.update(1)
    
    if not all_metrics:
        print("❌ No valid metrics calculated!")
        return
    
    # Create DataFrames
    metrics_df = pd.DataFrame(all_metrics)
    
    # Create output directory
    output_dir = "calibration_metrics"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    metrics_df.to_csv(os.path.join(output_dir, 'detailed_calibration_metrics.csv'), index=False)
    
    # Aggregate metrics per model
    print("📈 Aggregating metrics per model...")
    aggregated_df = aggregate_model_metrics(metrics_df)
    
    # Save aggregated results
    aggregated_df.to_csv(os.path.join(output_dir, 'model_averaged_calibration_metrics.csv'), index=False)
    
    # Save separate files for CoT and ZS
    cot_df = aggregated_df[aggregated_df['exp_type'] == 'cot_exp']
    zs_df = aggregated_df[aggregated_df['exp_type'] == 'zs_exp']
    
    if len(cot_df) > 0:
        cot_df.to_csv(os.path.join(output_dir, 'cot_calibration_metrics.csv'), index=False)
    
    if len(zs_df) > 0:
        zs_df.to_csv(os.path.join(output_dir, 'zs_calibration_metrics.csv'), index=False)
    
    # Create comparison plots
    print("📊 Creating calibration comparison plots...")
    create_calibration_comparison_plots(aggregated_df, output_dir)
    
    # Print summary statistics
    print(f"\n✅ Calibration metrics calculation complete!")
    print(f"   Successfully processed: {len(all_metrics)} files")
    print(f"   Failed files: {len(failed_files)}")
    print(f"   Unique models: {aggregated_df['model_name'].nunique()}")
    print(f"   CoT experiments: {len(cot_df)} models")
    print(f"   ZS experiments: {len(zs_df)} models")
    
    if failed_files:
        print("   Failed files:")
        for f in failed_files[:5]:  # Show first 5 failed files
            print(f"     {f}")
        if len(failed_files) > 5:
            print(f"     ... and {len(failed_files) - 5} more")
    
    print(f"\n📁 Results saved in '{output_dir}/' directory:")
    print(f"   📄 detailed_calibration_metrics.csv - Per file metrics")
    print(f"   📄 model_averaged_calibration_metrics.csv - Per model averages (both experiments)")
    print(f"   📄 cot_calibration_metrics.csv - Chain-of-Thought results only")
    print(f"   📄 zs_calibration_metrics.csv - Zero-Shot results only")
    print(f"   📊 calibration_metrics_comparison.pdf/png - CoT vs ZS comparison")
    print(f"   📊 calibration_metrics_rankings.pdf/png - Model rankings")
    
    # Print top performers
    if len(cot_df) > 0:
        best_cot_ece = cot_df.loc[cot_df['ece_avg'].idxmin()]
        best_cot_brier = cot_df.loc[cot_df['brier_avg'].idxmin()]
        print(f"\n🏆 Best CoT performers:")
        print(f"   ECE: {best_cot_ece['model_name']} ({best_cot_ece['ece_avg']:.4f})")
        print(f"   Brier: {best_cot_brier['model_name']} ({best_cot_brier['brier_avg']:.4f})")
    
    if len(zs_df) > 0:
        best_zs_ece = zs_df.loc[zs_df['ece_avg'].idxmin()]
        best_zs_brier = zs_df.loc[zs_df['brier_avg'].idxmin()]
        print(f"\n🏆 Best ZS performers:")
        print(f"   ECE: {best_zs_ece['model_name']} ({best_zs_ece['ece_avg']:.4f})")
        print(f"   Brier: {best_zs_brier['model_name']} ({best_zs_brier['brier_avg']:.4f})")

if __name__ == "__main__":
    main() 