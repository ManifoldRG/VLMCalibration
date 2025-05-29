import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
import os
import json
import glob
from pathlib import Path
import seaborn as sns
from typing import Dict, List, Tuple
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
warnings.filterwarnings('ignore')


MODEL_NAME_MAPPING = {
    'Llama-32-1B-Instruct': 'Llama 3.2 1B Instruct',
    'Llama-32-3B-Instruct': 'Llama 3.2 3B Instruct',
    'Meta-Llama-31-8B-Instruct': 'Llama 3.1 8B Instruct',

    'gemma-2-2b-it': 'Gemma 2 2B Instruct',
    'gemma-2-9b-it': 'Gemma 2 9B Instruct',

    'OLMo-2-1124-7B-Instruct': 'OLMo 2-1124 7B Instruct',
    'OLMo-2-1124-13B-Instruct': 'OLMo 2-1124 13B Instruct',

    'Qwen25-05B-Instruct': 'Qwen 2.5 0.5B Instruct',
    'Qwen25-3B-Instruct': 'Qwen 2.5 3B Instruct',
    'Qwen25-7B-Instruct': 'Qwen 2.5 7B Instruct',
    'Qwen25-14B-Instruct': 'Qwen 2.5 14B Instruct',
}

# Experiment type name mapping
EXP_TYPE_MAPPING = {
    'cot_exp': 'CoT Self Ask',
    'zs_exp': 'Direct Self Ask',
}

# Standardized plot sizes
INDIVIDUAL_PLOT_SIZE = (8, 6)  # Standardized size for per-model-dataset-split plots
SUMMARY_PLOT_SIZE = (12, 8)    # Size for summary/comparison plots
LARGE_PLOT_SIZE = (16, 10)     # Size for complex comparison plots

# Publication-quality matplotlib settings based on research best practices
def setup_publication_style():
    """Configure matplotlib for publication-quality figures following ICML/research paper standards"""
    
    # Font settings for publication (compatible with LaTeX/PDF)
    mpl.rcParams['pdf.fonttype'] = 42  # Required for Adobe Illustrator compatibility
    mpl.rcParams['ps.fonttype'] = 42   # Required for Adobe Illustrator compatibility
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    
    # Font sizes optimized for academic papers
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 10
    
    # Figure settings
    mpl.rcParams['figure.figsize'] = INDIVIDUAL_PLOT_SIZE  # Standardized individual plot size
    mpl.rcParams['figure.dpi'] = 300  # High quality
    mpl.rcParams['savefig.dpi'] = 300
    mpl.rcParams['savefig.bbox'] = 'tight'
    mpl.rcParams['savefig.pad_inches'] = 0.1
    
    # Line and marker settings
    mpl.rcParams['lines.linewidth'] = 1.5
    mpl.rcParams['lines.markersize'] = 6
    
    # Grid and axes
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.linewidth'] = 0.8
    
    # Colors - use colorblind-friendly palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    mpl.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)

def apply_clean_style(ax):
    """Apply clean styling to axes following publication best practices"""
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Style remaining spines
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    # Tick styling
    ax.tick_params(axis='both', direction='out', length=4, width=0.8, 
                   colors='#333333', labelcolor='#333333')
    ax.tick_params(axis='y', left=True, right=False)
    ax.tick_params(axis='x', top=False, bottom=True)
    
    # Grid (subtle)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, color='#cccccc')
    ax.set_axisbelow(True)

def categorize_model_family(model_name: str) -> str:
    """Categorize model into family based on model name"""
    model_lower = model_name.lower()
    
    if 'llama' in model_lower:
        return 'Llama'
    elif 'gemma' in model_lower:
        return 'Gemma'
    elif 'qwen' in model_lower:
        return 'Qwen'
    elif 'olmo' in model_lower:
        return 'OLMo'
    else:
        return 'Other'

def discover_data_files(base_dirs: List[str], quant: bool = False) -> Dict[str, List[str]]:
    """Discover all JSON data files excluding experiment_details
    
    Args:
        base_dirs: List of base directories to search
        quant: If False (default), exclude GGUF quantized models. If True, include them.
    """
    data_files = {}
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
            
        # Find all JSON files that match the pattern
        pattern = os.path.join(base_dir, "*", "*_records_*.json")
        files = glob.glob(pattern)
        
        # Filter out experiment_details files
        files = [f for f in files if 'experiment_details' not in os.path.basename(f)]
        
        # Filter GGUF files based on quant flag
        if not quant:
            files = [f for f in files if 'gguf' not in os.path.basename(f).lower()]
        
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

def load_json_data(filepath: str, sample: bool = True) -> pd.DataFrame:
    """Load and process JSON data file
    
    Args:
        filepath: Path to the JSON file
        sample: If True, sample max 1000 records for plotting. If False, load all data.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Check if 'correct' column exists, if not try to derive it
        if 'correct' not in df.columns:
            # This might need adjustment based on your actual data structure
            # For now, assume we can derive it from answer comparison
            print(f"Warning: 'correct' column not found in {filepath}")
            return None
        
        # Sample 1000 records with deterministic random state for plotting
        if len(df) > 1500 and sample:
            print(f"Sampling 1000 records from {filepath}")
            df = df.sample(n=1500, random_state=42).reset_index(drop=True)
        elif not sample:
            print(f"Loading all {len(df)} records from {filepath}")
            
        return df
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def create_confidence_histogram(df: pd.DataFrame, metadata: Dict[str, str], 
                              density: bool = True, save_dir: str = None) -> plt.Figure:
    """Create publication-quality confidence histogram"""
    
    fig, ax = plt.subplots(figsize=INDIVIDUAL_PLOT_SIZE)
    
    # Calculate statistics
    correct_data = df[df['correct'] == True]['p_true']
    incorrect_data = df[df['correct'] == False]['p_true']
    
    mean_conf_correct = correct_data.mean()
    mean_conf_incorrect = incorrect_data.mean()
    
    # Create histogram
    bins = np.linspace(0, 1, 21)
    
    # Plot with publication-quality colors
    correct_color = '#2E8B57'    # Sea green
    incorrect_color = '#DC143C'  # Crimson
    
    ax.hist(correct_data, bins=bins, alpha=0.7, density=density,
            label=f"Correct (μ={mean_conf_correct:.3f})", 
            color=correct_color, edgecolor='white', linewidth=0.5)
    
    ax.hist(incorrect_data, bins=bins, alpha=0.7, density=density,
            label=f"Incorrect (μ={mean_conf_incorrect:.3f})", 
            color=incorrect_color, edgecolor='white', linewidth=0.5)
    
    # Styling
    apply_clean_style(ax)
    
    # Labels and title
    ax.set_xlabel('Confidence (p_true)', fontweight='normal')
    ylabel = 'Density' if density else 'Count'
    ax.set_ylabel(ylabel, fontweight='normal')
    
    # Create a clean title using model name mapping
    model_display = MODEL_NAME_MAPPING.get(metadata['model_name'], metadata['model_name'].replace('-', ' ').replace('_', ' '))
    dataset_display = metadata['dataset'].upper()
    split_display = metadata['split'].capitalize()
    
    title = f"{model_display}\n{dataset_display} {split_display}"
    ax.set_title(title, fontweight='normal', pad=15)
    
    # Legend
    legend = ax.legend(loc='upper left', frameon=True, fancybox=False, 
                      edgecolor='#cccccc', facecolor='white', framealpha=0.9)
    legend.get_frame().set_linewidth(0.5)
    
    # Set limits
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    
    # Better tick spacing
    ax.set_xticks(np.arange(0, 1.1, 0.2))
    
    plt.tight_layout()
    
    # Save if directory provided
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Add experiment type prefix to filename
        exp_prefix = "cot_" if metadata['exp_type'] == 'cot_exp' else "zs_"
        filename = f"{exp_prefix}confidence_histogram_{metadata['dataset']}_{metadata['split']}_{metadata['model_name']}.pdf"
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        
        # Also save PNG for viewing
        png_filepath = filepath.replace('.pdf', '.png')
        fig.savefig(png_filepath, dpi=300, bbox_inches='tight')
    
    return fig

def create_per_dataset_summary(summary_df: pd.DataFrame, save_dir: str):
    """Create per-dataset summary plots showing confidence when correct vs incorrect"""
    
    datasets = sorted(summary_df['dataset'].unique())
    
    if len(datasets) == 0:
        print("⚠️  No datasets found for per-dataset summary")
        return
    
    # Create a subdirectory for per-dataset plots
    dataset_dir = os.path.join(save_dir, 'per_dataset')
    os.makedirs(dataset_dir, exist_ok=True)
    
    exp_colors = {'cot_exp': '#2E8B57', 'zs_exp': '#DC143C'}  # Green for CoT, Red for ZS
    exp_names = {'cot_exp': 'CoT', 'zs_exp': 'ZS'}
    
    for dataset in datasets:
        dataset_data = summary_df[summary_df['dataset'] == dataset].copy()
        
        if len(dataset_data) == 0:
            continue
        
        # Create the plot
        fig, ax = plt.subplots(figsize=SUMMARY_PLOT_SIZE)
        
        # Get unique models and experiment types for this dataset
        models = sorted(dataset_data['model'].unique())
        exp_types = sorted(dataset_data['exp_type'].unique())
        
        # Create marker styles for models
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '8']
        model_marker_map = {model: markers[i % len(markers)] for i, model in enumerate(models)}
        
        # Plot each model-experiment combination
        for _, row in dataset_data.iterrows():
            color = exp_colors.get(row['exp_type'], '#888888')
            marker = model_marker_map[row['model']]
            
            ax.scatter(row['mean_conf_correct'], row['mean_conf_incorrect'], 
                      c=[color], marker=marker, s=100, alpha=0.8, 
                      edgecolors='black', linewidth=0.5)
        
        # Add diagonal line (perfect calibration would be mean_conf_correct = mean_conf_incorrect)
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])
        ]
        ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1, label='Equal Confidence')
        
        # Create legends
        # Experiment type legend (colors)
        exp_handles = []
        for exp_type in exp_types:
            if exp_type in exp_colors:
                handle = plt.Line2D([0], [0], marker='o', color='w', 
                                   markerfacecolor=exp_colors[exp_type], 
                                   markersize=10, alpha=0.8, markeredgecolor='black',
                                   markeredgewidth=0.5, label=exp_names[exp_type])
                exp_handles.append(handle)
        
        # Model legend (markers) - show full model names
        model_handles = []
        for model in models:
            # Clean up model name for display
            model_display = model.replace('-', ' ').replace('_', ' ')
            
            handle = plt.Line2D([0], [0], marker=model_marker_map[model], color='w',
                               markerfacecolor='gray', markersize=8, alpha=0.8,
                               markeredgecolor='black', markeredgewidth=0.5,
                               label=model_display)
            model_handles.append(handle)
        
        # Add legends
        exp_legend = ax.legend(handles=exp_handles, loc='upper left', 
                              title='Experiment Type', frameon=True, fancybox=False,
                              edgecolor='#cccccc', facecolor='white', 
                              framealpha=0.9, title_fontsize=10)
        exp_legend.get_frame().set_linewidth(0.5)
        
        # Add model legend
        model_legend = ax.legend(handles=model_handles, loc='center left', 
                                bbox_to_anchor=(1.05, 0.5), title='Models',
                                frameon=True, fancybox=False, edgecolor='#cccccc',
                                facecolor='white', framealpha=0.9, title_fontsize=10,
                                fontsize=9)
        model_legend.get_frame().set_linewidth(0.5)
        
        # Add experiment legend back (matplotlib removes previous legend)
        ax.add_artist(exp_legend)
        
        apply_clean_style(ax)
        ax.set_xlabel('Mean Confidence When Correct', fontweight='normal')
        ax.set_ylabel('Mean Confidence When Incorrect', fontweight='normal')
        ax.set_title(f'Model Confidence Calibration - {dataset.upper()}', fontweight='normal', pad=15)
        
        # Set equal aspect ratio and reasonable limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Adjust layout to make room for model legend
        plt.subplots_adjust(right=0.7)
        
        # Save the plot
        filename_base = f'calibration_summary_{dataset.lower()}'
        fig.savefig(os.path.join(dataset_dir, f'{filename_base}.pdf'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(dataset_dir, f'{filename_base}.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Create annotated version for better model identification
        fig2, ax2 = plt.subplots(figsize=SUMMARY_PLOT_SIZE)
        
        # Plot with model names as text annotations
        for _, row in dataset_data.iterrows():
            color = exp_colors.get(row['exp_type'], '#888888')
            ax2.scatter(row['mean_conf_correct'], row['mean_conf_incorrect'], 
                       c=[color], s=120, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Add model name as annotation
            model_short = row['model'].replace('-', ' ').replace('_', ' ')
            if len(model_short) > 15:
                model_short = model_short[:12] + '...'
            
            # Add experiment type to annotation
            exp_label = exp_names.get(row['exp_type'], row['exp_type'])
            annotation = f"{model_short} ({exp_label})"
            
            ax2.annotate(annotation, (row['mean_conf_correct'], row['mean_conf_incorrect']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8,
                        ha='left', va='bottom', alpha=0.8)
        
        # Add diagonal line
        lims = [
            np.min([ax2.get_xlim(), ax2.get_ylim()]),
            np.max([ax2.get_xlim(), ax2.get_ylim()])
        ]
        ax2.plot(lims, lims, 'k--', alpha=0.3, linewidth=1, label='Equal Confidence')
        
        # Add experiment type legend only
        ax2.legend(handles=exp_handles + [plt.Line2D([0], [0], color='k', linestyle='--', alpha=0.3, label='Equal Confidence')], 
                  loc='upper left', title='Experiment Type',
                  frameon=True, fancybox=False, edgecolor='#cccccc',
                  facecolor='white', framealpha=0.9, title_fontsize=10)
        
        apply_clean_style(ax2)
        ax2.set_xlabel('Mean Confidence When Correct', fontweight='normal')
        ax2.set_ylabel('Mean Confidence When Incorrect', fontweight='normal')
        ax2.set_title(f'Model Confidence Calibration - {dataset.upper()} (Annotated)', fontweight='normal', pad=15)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        fig2.savefig(os.path.join(dataset_dir, f'{filename_base}_annotated.pdf'), dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(dataset_dir, f'{filename_base}_annotated.png'), dpi=300, bbox_inches='tight')
        plt.close(fig2)
    
    # Save per-dataset summary statistics
    for dataset in datasets:
        dataset_data = summary_df[summary_df['dataset'] == dataset]
        if len(dataset_data) > 0:
            dataset_data.to_csv(os.path.join(dataset_dir, f'summary_statistics_{dataset.lower()}.csv'), index=False)
    
    print(f"📊 Created per-dataset summary plots:")
    for dataset in datasets:
        print(f"   • {dataset.upper()}:")
        print(f"     - calibration_summary_{dataset.lower()}.pdf/png")
        print(f"     - calibration_summary_{dataset.lower()}_annotated.pdf/png")
        print(f"     - summary_statistics_{dataset.lower()}.csv")

def create_cot_vs_zs_comparison(summary_df: pd.DataFrame, save_dir: str):
    """Create comparison plot between CoT and ZS showing top/bottom 5 models by improvement"""
    
    # Separate CoT and ZS data
    cot_data = summary_df[summary_df['exp_type'] == 'cot_exp'].copy()
    zs_data = summary_df[summary_df['exp_type'] == 'zs_exp'].copy()
    
    if len(cot_data) == 0 or len(zs_data) == 0:
        print("⚠️  Cannot create CoT vs ZS comparison - missing data for one experiment type")
        return
    
    # Create comparison metrics for each model-dataset combination
    comparison_data = []
    
    for _, cot_row in cot_data.iterrows():
        # Find matching ZS experiment
        zs_match = zs_data[
            (zs_data['model'] == cot_row['model']) & 
            (zs_data['dataset'] == cot_row['dataset']) &
            (zs_data['split'] == cot_row['split'])
        ]
        
        if len(zs_match) == 1:
            zs_row = zs_match.iloc[0]
            
            # Calculate improvements (CoT - ZS)
            accuracy_improvement = cot_row['accuracy'] - zs_row['accuracy']
            conf_gap_improvement = cot_row['confidence_gap'] - zs_row['confidence_gap']
            
            comparison_data.append({
                'model': cot_row['model'],
                'dataset': cot_row['dataset'],
                'split': cot_row['split'],
                'model_dataset': f"{cot_row['model']}_{cot_row['dataset']}",
                'accuracy_improvement': accuracy_improvement,
                'conf_gap_improvement': conf_gap_improvement,
                'cot_accuracy': cot_row['accuracy'],
                'zs_accuracy': zs_row['accuracy'],
                'cot_conf_gap': cot_row['confidence_gap'],
                'zs_conf_gap': zs_row['confidence_gap']
            })
    
    if len(comparison_data) == 0:
        print("⚠️  No matching CoT-ZS pairs found for comparison")
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by accuracy improvement and get top/bottom 5
    comparison_df_sorted = comparison_df.sort_values('accuracy_improvement', ascending=False)
    top_5 = comparison_df_sorted.head(5)
    bottom_5 = comparison_df_sorted.tail(5)
    
    # Combine for plotting
    plot_data = pd.concat([top_5, bottom_5])
    plot_data['category'] = ['Top 5'] * 5 + ['Bottom 5'] * 5
    
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=LARGE_PLOT_SIZE)
    
    # Plot 1: Accuracy Improvement
    colors = ['#2E8B57' if x > 0 else '#DC143C' for x in plot_data['accuracy_improvement']]
    bars1 = ax1.barh(range(len(plot_data)), plot_data['accuracy_improvement'], color=colors, alpha=0.7)
    
    # Customize labels for readability
    labels = []
    for _, row in plot_data.iterrows():
        model_short = row['model'].replace('-', ' ').replace('_', ' ')
        if len(model_short) > 15:
            model_short = model_short[:12] + '...'
        labels.append(f"{model_short}\n{row['dataset'].upper()}")
    
    ax1.set_yticks(range(len(plot_data)))
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel('Accuracy Improvement (CoT - ZS)', fontweight='normal')
    ax1.set_title(f'Model Performance: {EXP_TYPE_MAPPING["cot_exp"]} vs {EXP_TYPE_MAPPING["zs_exp"]}\nAccuracy Improvement', fontweight='normal', pad=15)
    ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars1, plot_data['accuracy_improvement'])):
        ax1.text(value + (0.01 if value >= 0 else -0.01), i, f'{value:.3f}', 
                ha='left' if value >= 0 else 'right', va='center', fontsize=8)
    
    # Add category separators
    ax1.axhline(y=4.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.text(ax1.get_xlim()[1] * 0.7, 2, 'Top 5\n(Most Improved)', 
             ha='center', va='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))
    ax1.text(ax1.get_xlim()[1] * 0.7, 7, 'Bottom 5\n(Least Improved)', 
             ha='center', va='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.3))
    
    apply_clean_style(ax1)
    
    # Plot 2: Confidence Gap Improvement
    colors2 = ['#2E8B57' if x > 0 else '#DC143C' for x in plot_data['conf_gap_improvement']]
    bars2 = ax2.barh(range(len(plot_data)), plot_data['conf_gap_improvement'], color=colors2, alpha=0.7)
    
    ax2.set_yticks(range(len(plot_data)))
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel('Confidence Gap Improvement (CoT - ZS)', fontweight='normal')
    ax2.set_title(f'Model Performance: {EXP_TYPE_MAPPING["cot_exp"]} vs {EXP_TYPE_MAPPING["zs_exp"]}\nConfidence Gap Improvement', fontweight='normal', pad=15)
    ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars2, plot_data['conf_gap_improvement'])):
        ax2.text(value + (0.005 if value >= 0 else -0.005), i, f'{value:.3f}', 
                ha='left' if value >= 0 else 'right', va='center', fontsize=8)
    
    # Add category separators
    ax2.axhline(y=4.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    apply_clean_style(ax2)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'cot_vs_zs_comparison.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, 'cot_vs_zs_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save the comparison data
    comparison_df.to_csv(os.path.join(save_dir, 'cot_vs_zs_comparison_data.csv'), index=False)
    
    # Print summary statistics
    print(f"📊 CoT vs ZS Comparison Summary:")
    print(f"   • Models analyzed: {len(comparison_df)}")
    print(f"   • Average accuracy improvement: {comparison_df['accuracy_improvement'].mean():.3f}")
    print(f"   • Average confidence gap improvement: {comparison_df['conf_gap_improvement'].mean():.3f}")
    print(f"   • Models improved by CoT (accuracy): {(comparison_df['accuracy_improvement'] > 0).sum()}/{len(comparison_df)}")
    print(f"   • Best improvement: {comparison_df.loc[comparison_df['accuracy_improvement'].idxmax(), 'model_dataset']} (+{comparison_df['accuracy_improvement'].max():.3f})")
    print(f"   • Worst change: {comparison_df.loc[comparison_df['accuracy_improvement'].idxmin(), 'model_dataset']} ({comparison_df['accuracy_improvement'].min():.3f})")

def create_family_calibration_plots(summary_df: pd.DataFrame, save_dir: str):
    """Create calibration plots grouped by model families for each dataset and experiment type"""
    
    # Add family column to summary data
    summary_df['family'] = summary_df['model'].apply(categorize_model_family)
    
    # Create family plots directory
    family_dir = os.path.join(save_dir, 'family_plots')
    os.makedirs(family_dir, exist_ok=True)
    
    # Get unique datasets and experiment types
    datasets = sorted(summary_df['dataset'].unique())
    exp_types = sorted(summary_df['exp_type'].unique())
    
    # Color palette for families
    family_colors = {
        'Llama': '#1f77b4',    # Blue
        'Gemma': '#ff7f0e',    # Orange  
        'Qwen': '#2ca02c',     # Green
        'OLMo': '#d62728',     # Red
        'Other': '#9467bd'     # Purple
    }
    
    for dataset in datasets:
        for exp_type in exp_types:
            # Filter data for this dataset and experiment type
            data = summary_df[
                (summary_df['dataset'] == dataset) & 
                (summary_df['exp_type'] == exp_type)
            ].copy()
            
            if len(data) == 0:
                continue
            
            # Get families present in this data
            families = sorted(data['family'].unique())
            
            if len(families) == 0:
                continue
            
            # Determine experiment name for titles and filenames
            exp_name = EXP_TYPE_MAPPING[exp_type]
            exp_short = "cot" if exp_type == "cot_exp" else "zs"
            
            # Create the plot
            fig, ax = plt.subplots(figsize=SUMMARY_PLOT_SIZE)
            
            # Plot each family
            for family in families:
                family_data = data[data['family'] == family]
                
                if len(family_data) == 0:
                    continue
                
                color = family_colors.get(family, '#888888')
                
                # Plot scatter points for each model in the family
                ax.scatter(family_data['mean_conf_correct'], 
                          family_data['mean_conf_incorrect'],
                          c=[color], s=80, alpha=0.7, 
                          label=f'{family} (n={len(family_data)})',
                          edgecolors='black', linewidth=0.5)
                
                # Add family mean as a larger marker
                family_mean_correct = family_data['mean_conf_correct'].mean()
                family_mean_incorrect = family_data['mean_conf_incorrect'].mean()
                
                ax.scatter(family_mean_correct, family_mean_incorrect,
                          c=[color], s=200, alpha=0.9, marker='D',
                          edgecolors='black', linewidth=1.5)
                
                # Add family name annotation near the mean
                ax.annotate(f'{family}\nMean', 
                           (family_mean_correct, family_mean_incorrect),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=9, fontweight='bold', ha='left', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor=color, alpha=0.3))
            
            # Add diagonal line (perfect calibration)
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),
                np.max([ax.get_xlim(), ax.get_ylim()])
            ]
            ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1, 
                   label='Equal Confidence')
            
            # Styling
            apply_clean_style(ax)
            ax.set_xlabel('Mean Confidence When Correct', fontweight='normal')
            ax.set_ylabel('Mean Confidence When Incorrect', fontweight='normal')
            ax.set_title(f'Family Calibration - {dataset.upper()} ({exp_name})', 
                        fontweight='normal', pad=15)
            
            # Set limits (remove forced equal aspect ratio)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            
            # Legend
            legend = ax.legend(loc='upper left', frameon=True, fancybox=False,
                              edgecolor='#cccccc', facecolor='white', 
                              framealpha=0.9, fontsize=10)
            legend.get_frame().set_linewidth(0.5)
            
            plt.tight_layout()
            
            # Save the plot
            filename_base = f'family_calibration_{dataset.lower()}_{exp_short}'
            fig.savefig(os.path.join(family_dir, f'{filename_base}.pdf'), 
                       dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(family_dir, f'{filename_base}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Create family statistics summary
            family_stats = []
            for family in families:
                family_data = data[data['family'] == family]
                if len(family_data) > 0:
                    family_stats.append({
                        'family': family,
                        'dataset': dataset,
                        'exp_type': exp_type,
                        'n_models': len(family_data),
                        'mean_accuracy': family_data['accuracy'].mean(),
                        'std_accuracy': family_data['accuracy'].std(),
                        'mean_conf_correct': family_data['mean_conf_correct'].mean(),
                        'std_conf_correct': family_data['mean_conf_correct'].std(),
                        'mean_conf_incorrect': family_data['mean_conf_incorrect'].mean(),
                        'std_conf_incorrect': family_data['mean_conf_incorrect'].std(),
                        'mean_conf_gap': family_data['confidence_gap'].mean(),
                        'std_conf_gap': family_data['confidence_gap'].std()
                    })
            
            # Save family statistics
            if family_stats:
                family_stats_df = pd.DataFrame(family_stats)
                stats_filename = f'family_statistics_{dataset.lower()}_{exp_short}.csv'
                family_stats_df.to_csv(os.path.join(family_dir, stats_filename), index=False)
    
    # Create overall family comparison across all datasets
    create_overall_family_comparison(summary_df, family_dir, family_colors)
    
    print(f"📊 Created family calibration plots:")
    for dataset in datasets:
        for exp_type in exp_types:
            exp_short = "cot" if exp_type == "cot_exp" else "zs"
            exp_name = EXP_TYPE_MAPPING[exp_type]
            print(f"   • {dataset.upper()} ({exp_name}):")
            print(f"     - family_calibration_{dataset.lower()}_{exp_short}.pdf/png")
            print(f"     - family_statistics_{dataset.lower()}_{exp_short}.csv")

def create_overall_family_comparison(summary_df: pd.DataFrame, save_dir: str, family_colors: Dict[str, str]):
    """Create overall family comparison plots across all datasets"""
    
    # Separate by experiment type
    exp_types = sorted(summary_df['exp_type'].unique())
    
    for exp_type in exp_types:
        exp_data = summary_df[summary_df['exp_type'] == exp_type].copy()
        
        if len(exp_data) == 0:
            continue
        
        exp_name = EXP_TYPE_MAPPING[exp_type]
        exp_short = "cot" if exp_type == "cot_exp" else "zs"
        
        # Create family aggregated statistics
        family_agg = exp_data.groupby(['family', 'dataset']).agg({
            'accuracy': ['mean', 'std', 'count'],
            'mean_conf_correct': ['mean', 'std'],
            'mean_conf_incorrect': ['mean', 'std'],
            'confidence_gap': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        family_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                             for col in family_agg.columns]
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=LARGE_PLOT_SIZE)
        
        # Plot 1: Accuracy by Family
        families = sorted(family_agg['family'].unique())
        datasets = sorted(family_agg['dataset'].unique())
        
        x_pos = np.arange(len(families))
        width = 0.8 / len(datasets)
        
        for i, dataset in enumerate(datasets):
            dataset_data = family_agg[family_agg['dataset'] == dataset]
            
            accuracies = []
            errors = []
            colors = []
            
            for family in families:
                family_data = dataset_data[dataset_data['family'] == family]
                if len(family_data) > 0:
                    accuracies.append(family_data['accuracy_mean'].iloc[0])
                    errors.append(family_data['accuracy_std'].iloc[0] if not pd.isna(family_data['accuracy_std'].iloc[0]) else 0)
                    colors.append(family_colors.get(family, '#888888'))
                else:
                    accuracies.append(0)
                    errors.append(0)
                    colors.append('#cccccc')
            
            bars = ax1.bar(x_pos + i * width, accuracies, width, 
                          label=dataset.upper(), alpha=0.8,
                          yerr=errors, capsize=3, color=colors)
        
        ax1.set_xlabel('Model Family', fontweight='normal')
        ax1.set_ylabel('Accuracy', fontweight='normal')
        ax1.set_title(f'Family Performance Comparison - {exp_name}\nAccuracy by Dataset', 
                     fontweight='normal', pad=15)
        ax1.set_xticks(x_pos + width * (len(datasets) - 1) / 2)
        ax1.set_xticklabels(families)
        ax1.legend(title='Dataset', loc='upper left')
        ax1.set_ylim(0, 1)
        apply_clean_style(ax1)
        
        # Plot 2: Confidence Gap by Family
        for i, dataset in enumerate(datasets):
            dataset_data = family_agg[family_agg['dataset'] == dataset]
            
            conf_gaps = []
            errors = []
            colors = []
            
            for family in families:
                family_data = dataset_data[dataset_data['family'] == family]
                if len(family_data) > 0:
                    conf_gaps.append(family_data['confidence_gap_mean'].iloc[0])
                    errors.append(family_data['confidence_gap_std'].iloc[0] if not pd.isna(family_data['confidence_gap_std'].iloc[0]) else 0)
                    colors.append(family_colors.get(family, '#888888'))
                else:
                    conf_gaps.append(0)
                    errors.append(0)
                    colors.append('#cccccc')
            
            bars = ax2.bar(x_pos + i * width, conf_gaps, width,
                          label=dataset.upper(), alpha=0.8,
                          yerr=errors, capsize=3, color=colors)
        
        ax2.set_xlabel('Model Family', fontweight='normal')
        ax2.set_ylabel('Confidence Gap (Correct - Incorrect)', fontweight='normal')
        ax2.set_title(f'Family Performance Comparison - {exp_name}\nConfidence Gap by Dataset', 
                     fontweight='normal', pad=15)
        ax2.set_xticks(x_pos + width * (len(datasets) - 1) / 2)
        ax2.set_xticklabels(families)
        ax2.legend(title='Dataset', loc='upper left')
        apply_clean_style(ax2)
        
        plt.tight_layout()
        
        # Save the plot
        filename_base = f'overall_family_comparison_{exp_short}'
        fig.savefig(os.path.join(save_dir, f'{filename_base}.pdf'), 
                   dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(save_dir, f'{filename_base}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save aggregated family statistics
        family_agg.to_csv(os.path.join(save_dir, f'overall_family_statistics_{exp_short}.csv'), index=False)
    
    print(f"   • Overall Family Comparisons:")
    for exp_type in exp_types:
        exp_short = "cot" if exp_type == "cot_exp" else "zs"
        exp_name = EXP_TYPE_MAPPING[exp_type]
        print(f"     - overall_family_comparison_{exp_short}.pdf/png ({exp_name})")
        print(f"     - overall_family_statistics_{exp_short}.csv")

def create_summary_plots(all_data: Dict, save_dir: str = "summary_plots"):
    """Create summary plots across models and datasets, separated by experiment type"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Aggregate data for summary statistics
    summary_data = []
    
    for key, df_info in all_data.items():
        df, metadata = df_info
        if df is None:
            continue
            
        correct_data = df[df['correct'] == True]['p_true']
        incorrect_data = df[df['correct'] == False]['p_true']
        
        summary_data.append({
            'model': metadata['model_name'],
            'dataset': metadata['dataset'],
            'split': metadata['split'],
            'exp_type': metadata['exp_type'],
            'accuracy': (df['correct'] == True).mean(),
            'mean_conf_correct': correct_data.mean(),
            'mean_conf_incorrect': incorrect_data.mean(),
            'confidence_gap': correct_data.mean() - incorrect_data.mean(),
            'n_samples': len(df)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create separate plots for CoT and ZS experiments
    exp_types = summary_df['exp_type'].unique()
    
    for exp_type in exp_types:
        exp_data = summary_df[summary_df['exp_type'] == exp_type]
        
        if len(exp_data) == 0:
            continue
            
        # Determine experiment name for titles and filenames
        exp_name = EXP_TYPE_MAPPING[exp_type]
        exp_short = "cot" if exp_type == "cot_exp" else "zs"
        
        # Create accuracy vs confidence gap plot with proper model labeling
        fig, ax = plt.subplots(figsize=SUMMARY_PLOT_SIZE)
        
        # Get unique datasets and models for this experiment type
        datasets = sorted(exp_data['dataset'].unique())
        models = sorted(exp_data['model'].unique())
        
        # Create color palette for datasets
        dataset_colors = plt.cm.Set1(np.linspace(0, 1, len(datasets)))
        dataset_color_map = {dataset: dataset_colors[i] for i, dataset in enumerate(datasets)}
        
        # Create marker styles for models
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '8']
        model_marker_map = {model: markers[i % len(markers)] for i, model in enumerate(models)}
        
        # Plot each combination
        for _, row in exp_data.iterrows():
            color = dataset_color_map[row['dataset']]
            marker = model_marker_map[row['model']]
            
            ax.scatter(row['accuracy'], row['confidence_gap'], 
                      c=[color], marker=marker, s=80, alpha=0.8, 
                      edgecolors='black', linewidth=0.5)
        
        # Create legends
        # Dataset legend (colors)
        dataset_handles = []
        for dataset in datasets:
            handle = plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=dataset_color_map[dataset], 
                               markersize=8, alpha=0.8, markeredgecolor='black',
                               markeredgewidth=0.5, label=dataset.upper())
            dataset_handles.append(handle)
        
        # Model legend (markers) - show full model names
        model_handles = []
        for model in models:
            # Clean up model name for display but keep full length
            model_display = model.replace('-', ' ').replace('_', ' ')
            
            handle = plt.Line2D([0], [0], marker=model_marker_map[model], color='w',
                               markerfacecolor='gray', markersize=8, alpha=0.8,
                               markeredgecolor='black', markeredgewidth=0.5,
                               label=model_display)
            model_handles.append(handle)
        
        # Add legends
        dataset_legend = ax.legend(handles=dataset_handles, loc='upper left', 
                                  title='Datasets', frameon=True, fancybox=False,
                                  edgecolor='#cccccc', facecolor='white', 
                                  framealpha=0.9, title_fontsize=10)
        dataset_legend.get_frame().set_linewidth(0.5)
        
        # Add model legend with more space for full names
        model_legend = ax.legend(handles=model_handles, loc='center left', 
                                bbox_to_anchor=(1.05, 0.5), title='Models',
                                frameon=True, fancybox=False, edgecolor='#cccccc',
                                facecolor='white', framealpha=0.9, title_fontsize=10,
                                fontsize=9)
        model_legend.get_frame().set_linewidth(0.5)
        
        # Add dataset legend back (matplotlib removes previous legend)
        ax.add_artist(dataset_legend)
        
        apply_clean_style(ax)
        ax.set_xlabel('Accuracy', fontweight='normal')
        ax.set_ylabel('Confidence Gap (Correct - Incorrect)', fontweight='normal')
        ax.set_title(f'Model Calibration Overview - {exp_name}', fontweight='normal', pad=15)
        
        # Set reasonable axis limits
        ax.set_xlim(-0.05, 1.05)
        
        # Adjust layout to make room for full model names
        plt.subplots_adjust(right=0.7)
        fig.savefig(os.path.join(save_dir, f'{exp_short}_calibration_overview.pdf'), dpi=300, bbox_inches='tight')
        fig.savefig(os.path.join(save_dir, f'{exp_short}_calibration_overview.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Create additional model-specific plot for better readability
        fig2, ax2 = plt.subplots(figsize=SUMMARY_PLOT_SIZE)
        
        # Plot with model names as text annotations
        for _, row in exp_data.iterrows():
            color = dataset_color_map[row['dataset']]
            ax2.scatter(row['accuracy'], row['confidence_gap'], 
                       c=[color], s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Add model name as annotation (shortened)
            model_short = row['model'].replace('-', ' ').replace('_', ' ')
            if len(model_short) > 15:
                model_short = model_short[:12] + '...'
            
            ax2.annotate(model_short, (row['accuracy'], row['confidence_gap']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8,
                        ha='left', va='bottom', alpha=0.8)
        
        # Add dataset legend only
        ax2.legend(handles=dataset_handles, loc='upper left', title='Datasets',
                  frameon=True, fancybox=False, edgecolor='#cccccc',
                  facecolor='white', framealpha=0.9, title_fontsize=10)
        
        apply_clean_style(ax2)
        ax2.set_xlabel('Accuracy', fontweight='normal')
        ax2.set_ylabel('Confidence Gap (Correct - Incorrect)', fontweight='normal')
        ax2.set_title(f'Model Calibration Overview - {exp_name} (with Model Names)', fontweight='normal', pad=15)
        ax2.set_xlim(-0.05, 1.05)
        
        plt.tight_layout()
        fig2.savefig(os.path.join(save_dir, f'{exp_short}_calibration_overview_annotated.pdf'), dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(save_dir, f'{exp_short}_calibration_overview_annotated.png'), dpi=300, bbox_inches='tight')
        plt.close(fig2)
    
    # Create CoT vs ZS comparison plot
    create_cot_vs_zs_comparison(summary_df, save_dir)
    
    # Create per-dataset summary plots
    create_per_dataset_summary(summary_df, save_dir)
    
    # Create family calibration plots
    create_family_calibration_plots(summary_df, save_dir)
    
    # Save summary statistics (still combined for comparison)
    summary_df.to_csv(os.path.join(save_dir, 'summary_statistics.csv'), index=False)
    
    # Also save separate CSV files for each experiment type
    for exp_type in exp_types:
        exp_data = summary_df[summary_df['exp_type'] == exp_type]
        if len(exp_data) > 0:
            exp_short = "cot" if exp_type == "cot_exp" else "zs"
            exp_data.to_csv(os.path.join(save_dir, f'summary_statistics_{exp_short}.csv'), index=False)
    
    print(f"📊 Created separate summary plots for each experiment type:")
    for exp_type in exp_types:
        exp_short = "cot" if exp_type == "cot_exp" else "zs"
        exp_name = EXP_TYPE_MAPPING[exp_type]
        print(f"   • {exp_name} ({exp_short.upper()}):")
        print(f"     - {exp_short}_calibration_overview.pdf/png")
        print(f"     - {exp_short}_calibration_overview_annotated.pdf/png")
        print(f"     - summary_statistics_{exp_short}.csv")
    print(f"   • CoT vs ZS Comparison:")
    print(f"     - cot_vs_zs_comparison.pdf/png")
    print(f"     - cot_vs_zs_comparison_data.csv")
    print(f"   • Combined: summary_statistics.csv")

def process_data_only(filepath: str, sample: bool = True) -> Tuple[str, tuple, bool]:
    """Process a single data file for summary statistics without generating plots"""
    try:
        metadata = parse_filename(filepath)
        
        # Skip train files
        if metadata['split'] == 'train':
            return filepath, None, False
        
        df = load_json_data(filepath, sample=sample)
        if df is None:
            return filepath, None, False
        
        # Create key for all_data dictionary
        key = f"{metadata['exp_type']}_{metadata['dataset']}_{metadata['split']}_{metadata['model_name']}"
        
        return filepath, (key, (df, metadata)), True
        
    except Exception as e:
        return filepath, None, False

def process_single_file(filepath: str, sample: bool = True) -> Tuple[str, tuple, bool]:
    """Process a single data file and generate plots"""
    try:
        metadata = parse_filename(filepath)
        
        # Skip train files
        if metadata['split'] == 'train':
            return filepath, None, False
        
        df = load_json_data(filepath, sample=sample)
        if df is None:
            return filepath, None, False
        
        # Create output directories for density and raw plots
        exp_short = "cot" if metadata['exp_type'] == "cot_exp" else "zs"
        density_dir = os.path.join("plots", exp_short, "density", metadata['dataset'])
        raw_dir = os.path.join("plots", exp_short, "raw", metadata['dataset'])
        
        # Generate density plot
        fig_density = create_confidence_histogram(df, metadata, density=True, save_dir=density_dir)
        plt.close(fig_density)
        
        # Generate raw count plot
        fig_raw = create_confidence_histogram(df, metadata, density=False, save_dir=raw_dir)
        plt.close(fig_raw)
        
        # Create key for all_data dictionary
        key = f"{metadata['exp_type']}_{metadata['dataset']}_{metadata['split']}_{metadata['model_name']}"
        
        return filepath, (key, (df, metadata)), True
        
    except Exception as e:
        return filepath, None, False

def list_available_models(base_dirs: List[str], quant: bool = False) -> List[str]:
    """List all available models in the data files
    
    Args:
        base_dirs: List of base directories to search
        quant: If False (default), exclude GGUF quantized models. If True, include them.
    
    Returns:
        List of unique model names found in the data
    """
    data_files = discover_data_files(base_dirs, quant=quant)
    
    if not data_files:
        return []
    
    # Collect all file paths
    all_files = []
    for base_dir, files in data_files.items():
        all_files.extend(files)
    
    # Extract model names from filenames
    models = set()
    for filepath in all_files:
        try:
            metadata = parse_filename(filepath)
            # Skip train files
            if metadata['split'] == 'train':
                continue
            models.add(metadata['model_name'])
        except Exception:
            continue
    
    return sorted(list(models))

def show_help():
    """Display help information for command line usage"""
    print("📊 VLM Calibration Plot Generator")
    print("=" * 50)
    print()
    print("USAGE:")
    print("  python generate_all_plots.py [OPTIONS]")
    print()
    print("OPTIONS:")
    print("  -h, --help           Show this help message")
    print("  -s, --summary-only   Generate only summary plots (no individual histograms)")
    print("  -f, --family-only    Generate only family analysis plots")
    print("  -m, --model MODEL    Generate plots for a specific model only")
    print("  -l, --list-models    List all available models and exit")
    print("  -q, --quant          Include GGUF quantized models (excluded by default)")
    print("  --no-sample          Load all data (default: sample max 1000 records per file)")
    print()
    print("EXAMPLES:")
    print("  # Generate all plots (default)")
    print("  python generate_all_plots.py")
    print()
    print("  # Generate only summary plots")
    print("  python generate_all_plots.py --summary-only")
    print()
    print("  # Generate family analysis plots only")
    print("  python generate_all_plots.py --family-only")
    print()
    print("  # Analyze a specific model")
    print("  python generate_all_plots.py --model llama-2-7b-chat")
    print()
    print("  # List available models")
    print("  python generate_all_plots.py --list-models")
    print()
    print("  # Include quantized models")
    print("  python generate_all_plots.py --quant")
    print()
    print("  # Load all data without sampling")
    print("  python generate_all_plots.py --no-sample")
    print()
    print("OUTPUT DIRECTORIES:")
    print("  📂 plots/                    Individual confidence histograms")
    print("  📂 summary_plots/            Summary and comparison plots")
    print("  📂 family_analysis/          Family-wise analysis plots")
    print("  📂 single_model_analysis/    Single model analysis plots")
    print()
    print("💡 All plots are saved in both PDF (vector) and PNG (raster) formats")

def main(summary_only: bool = False, quant: bool = False, family_only: bool = False, 
         sample: bool = True, model_name: str = None):
    """Main function to generate all plots
    
    Args:
        summary_only: If True, only generate summary plots without individual confidence histograms
        quant: If False (default), exclude GGUF quantized models. If True, include them.
        family_only: If True, only generate family analysis plots
        sample: If True (default), sample max 1000 records per file. If False, load all data.
        model_name: If provided, only generate plots for this specific model
    """
    
    # Setup publication style
    setup_publication_style()
    
    if sample:
        print("🎲 Using deterministic sampling: max 1000 records per file (random_state=42)")
    else:
        print("📊 Loading all data (no sampling)")
    
    if not quant:
        print("🚫 Excluding GGUF quantized models (use --quant to include them)")
    else:
        print("✅ Including GGUF quantized models")
    
    # Discover all data files
    base_dirs = ['cot_exp', 'zs_exp']
    data_files = discover_data_files(base_dirs, quant=quant)
    
    if not data_files:
        print("❌ No data files found!")
        return
    
    # Collect all file paths
    all_files = []
    for base_dir, files in data_files.items():
        all_files.extend(files)
    
    print(f"📊 Found {len(all_files)} data files across {len(data_files)} experiment types")
    
    # Choose processing function based on mode
    process_func = process_data_only if (summary_only or family_only or model_name) else process_single_file
    mode_description = "Processing data for analysis" if (summary_only or family_only or model_name) else "Processing files"
    
    # Process all files with ThreadPoolExecutor
    all_data = {}
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all tasks with sample parameter
        future_to_file = {executor.submit(process_func, filepath, sample): filepath 
                         for filepath in all_files}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(all_files), desc=mode_description, unit="file") as pbar:
            for future in as_completed(future_to_file):
                filepath = future_to_file[future]
                try:
                    result_filepath, data_result, success = future.result()
                    
                    if success and data_result is not None:
                        key, (df, metadata) = data_result
                        all_data[key] = (df, metadata)
                        pbar.set_postfix_str(f"✅ {os.path.basename(filepath)}")
                    else:
                        failed_files.append(result_filepath)
                        pbar.set_postfix_str(f"❌ {os.path.basename(filepath)}")
                        
                except Exception as e:
                    failed_files.append(filepath)
                    pbar.set_postfix_str(f"❌ {os.path.basename(filepath)} - {str(e)}")
                
                pbar.update(1)
    
    # Generate plots based on mode
    if model_name:
        print(f"🤖 Creating single model analysis plots for: {model_name}")
        plots_for_single_model(all_data, model_name, quant=quant)
    elif family_only:
        print("🏠 Creating family analysis plots...")
        plots_across_families(all_data, quant=quant)
    else:
        # Create summary plots
        print("📈 Creating summary plots...")
        create_summary_plots(all_data)
    
    # Print summary
    print(f"\n✅ Processing complete!")
    print(f"   Successfully processed: {len(all_data)} files")
    print(f"   Failed files: {len(failed_files)}")
    
    if failed_files:
        print("   Failed files:")
        for f in failed_files:
            print(f"     {f}")
    
    if model_name:
        print(f"\n📁 Single model analysis plots saved in 'single_model_analysis/' directory")
    elif family_only:
        print(f"\n📁 Family analysis plots saved in 'family_analysis/' directory")
    elif summary_only:
        print(f"\n📁 Summary plots saved in 'summary_plots/' directory")
    else:
        print(f"\n📁 Plots saved in organized structure:")
        print(f"   📂 plots/cot/density/{'{dataset}'}/")
        print(f"   📂 plots/cot/raw/{'{dataset}'}/")
        print(f"   📂 plots/zs/density/{'{dataset}'}/")
        print(f"   📂 plots/zs/raw/{'{dataset}'}/")
        print(f"📁 Summary plots saved in 'summary_plots/' directory")
    
    print(f"💡 All plots are saved in both PDF (vector) and PNG (raster) formats")

def plots_across_families(all_data: Dict, save_dir: str = "family_analysis", quant: bool = False):
    """Create focused family-wise confidence distribution plots for each task/dataset
    
    Args:
        all_data: Dictionary containing processed data
        save_dir: Directory to save family analysis plots
        quant: Whether quantized models are included in the analysis
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Get unique datasets and experiment types from the data
    datasets = set()
    exp_types = set()
    
    for key, df_info in all_data.items():
        df, metadata = df_info
        if df is not None:
            datasets.add(metadata['dataset'])
            exp_types.add(metadata['exp_type'])
    
    datasets = sorted(datasets)
    exp_types = sorted(exp_types)
    
    # Color palette for families
    family_colors = {
        'Llama': '#1f77b4',    # Blue
        'Gemma': '#ff7f0e',    # Orange  
        'Qwen': '#2ca02c',     # Green
        'OLMo': '#d62728',     # Red
        'Other': '#9467bd'     # Purple
    }
    
    print(f"🏠 Creating family-wise confidence distribution analysis...")
    print(f"   Datasets: {', '.join(datasets)}")
    print(f"   Experiment types: {', '.join(['CoT' if exp == 'cot_exp' else 'ZS' for exp in exp_types])}")
    print(f"   Quantized models: {'Included' if quant else 'Excluded'}")
    
    # Create confidence distribution plots for each dataset and experiment type
    created_plots = []
    
    for dataset in datasets:
        for exp_type in exp_types:
            print(f"   📊 Processing {dataset.upper()} - {'CoT' if exp_type == 'cot_exp' else 'ZS'}...")
            
            # Create confidence distribution plots for this dataset/experiment
            result = create_family_confidence_distributions(
                all_data, dataset, exp_type, family_colors, save_dir
            )
            
            if result:
                filename_correct, filename_incorrect, filename_combined = result
                created_plots.extend([
                    (dataset, exp_type, 'correct', filename_correct),
                    (dataset, exp_type, 'incorrect', filename_incorrect),
                    (dataset, exp_type, 'combined', filename_combined)
                ])
    
    # Create overall summary statistics across all tasks
    create_family_summary_statistics(all_data, save_dir, family_colors, quant)
    
    print(f"📊 Family confidence distribution analysis complete!")
    print(f"   Results saved in '{save_dir}/' directory:")
    
    # Group and display created plots
    for dataset in datasets:
        for exp_type in exp_types:
            exp_short = "cot" if exp_type == "cot_exp" else "zs"
            exp_name = EXP_TYPE_MAPPING[exp_type]
            print(f"   • {dataset.upper()} ({exp_name}):")
            print(f"     - family_confidence_correct_{dataset.lower()}_{exp_short}.pdf/png")
            print(f"     - family_confidence_incorrect_{dataset.lower()}_{exp_short}.pdf/png")
            print(f"     - family_confidence_combined_{dataset.lower()}_{exp_short}.pdf/png")
            print(f"     - family_confidence_stats_{dataset.lower()}_{exp_short}.csv")
    
    print(f"   • Overall Analysis:")
    print(f"     - family_summary_statistics.csv")
    print(f"     - family_confidence_overview.pdf/png")

def create_family_summary_statistics(all_data: Dict, save_dir: str, 
                                   family_colors: Dict[str, str], quant: bool):
    """Create overall summary statistics and overview plot for family analysis"""
    
    # Aggregate data for family analysis
    summary_data = []
    
    for key, df_info in all_data.items():
        df, metadata = df_info
        if df is None:
            continue
            
        correct_data = df[df['correct'] == True]['p_true']
        incorrect_data = df[df['correct'] == False]['p_true']
        
        summary_data.append({
            'model': metadata['model_name'],
            'dataset': metadata['dataset'],
            'split': metadata['split'],
            'exp_type': metadata['exp_type'],
            'family': categorize_model_family(metadata['model_name']),
            'accuracy': (df['correct'] == True).mean(),
            'mean_conf_correct': correct_data.mean(),
            'mean_conf_incorrect': incorrect_data.mean(),
            'confidence_gap': correct_data.mean() - incorrect_data.mean(),
            'n_samples': len(df)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    if len(summary_df) == 0:
        print("❌ No data available for family summary statistics")
        return
    
    # Save comprehensive statistics
    comprehensive_stats = summary_df.groupby(['family', 'dataset', 'exp_type']).agg({
        'accuracy': ['mean', 'std', 'count'],
        'confidence_gap': ['mean', 'std'],
        'mean_conf_correct': ['mean', 'std'],
        'mean_conf_incorrect': ['mean', 'std']
    }).round(4)
    
    comprehensive_stats.to_csv(os.path.join(save_dir, 'family_summary_statistics.csv'))
    
    # Create overview plot showing family performance across datasets
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Family Confidence Analysis Overview', fontsize=16, fontweight='bold')
    
    datasets = sorted(summary_df['dataset'].unique())
    families = sorted(summary_df['family'].unique())
    
    # Plot 1: Mean confidence when correct by family and dataset (CoT)
    cot_data = summary_df[summary_df['exp_type'] == 'cot_exp']
    if len(cot_data) > 0:
        conf_correct_matrix_cot = cot_data.groupby(['family', 'dataset'])['mean_conf_correct'].mean().unstack(fill_value=0)
        im1 = axes[0,0].imshow(conf_correct_matrix_cot.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[0,0].set_xticks(range(len(conf_correct_matrix_cot.columns)))
        axes[0,0].set_xticklabels(conf_correct_matrix_cot.columns, rotation=45)
        axes[0,0].set_yticks(range(len(conf_correct_matrix_cot.index)))
        axes[0,0].set_yticklabels(conf_correct_matrix_cot.index)
        axes[0,0].set_title(f'{EXP_TYPE_MAPPING["cot_exp"]}\nMean Confidence When Correct', fontweight='bold')
        
        # Add text annotations
        for i in range(len(conf_correct_matrix_cot.index)):
            for j in range(len(conf_correct_matrix_cot.columns)):
                text = axes[0,0].text(j, i, f'{conf_correct_matrix_cot.iloc[i, j]:.3f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im1, ax=axes[0,0], fraction=0.046, pad=0.04)
    
    # Plot 2: Mean confidence when correct by family and dataset (ZS)
    zs_data = summary_df[summary_df['exp_type'] == 'zs_exp']
    if len(zs_data) > 0:
        conf_correct_matrix_zs = zs_data.groupby(['family', 'dataset'])['mean_conf_correct'].mean().unstack(fill_value=0)
        im2 = axes[0,1].imshow(conf_correct_matrix_zs.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        axes[0,1].set_xticks(range(len(conf_correct_matrix_zs.columns)))
        axes[0,1].set_xticklabels(conf_correct_matrix_zs.columns, rotation=45)
        axes[0,1].set_yticks(range(len(conf_correct_matrix_zs.index)))
        axes[0,1].set_yticklabels(conf_correct_matrix_zs.index)
        axes[0,1].set_title(f'{EXP_TYPE_MAPPING["zs_exp"]}\nMean Confidence When Correct', fontweight='bold')
        
        # Add text annotations
        for i in range(len(conf_correct_matrix_zs.index)):
            for j in range(len(conf_correct_matrix_zs.columns)):
                text = axes[0,1].text(j, i, f'{conf_correct_matrix_zs.iloc[i, j]:.3f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im2, ax=axes[0,1], fraction=0.046, pad=0.04)
    
    # Plot 3: Mean confidence when incorrect by family and dataset (CoT)
    if len(cot_data) > 0:
        conf_incorrect_matrix_cot = cot_data.groupby(['family', 'dataset'])['mean_conf_incorrect'].mean().unstack(fill_value=0)
        im3 = axes[1,0].imshow(conf_incorrect_matrix_cot.values, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        axes[1,0].set_xticks(range(len(conf_incorrect_matrix_cot.columns)))
        axes[1,0].set_xticklabels(conf_incorrect_matrix_cot.columns, rotation=45)
        axes[1,0].set_yticks(range(len(conf_incorrect_matrix_cot.index)))
        axes[1,0].set_yticklabels(conf_incorrect_matrix_cot.index)
        axes[1,0].set_title(f'{EXP_TYPE_MAPPING["cot_exp"]}\nMean Confidence When Incorrect', fontweight='bold')
        
        # Add text annotations
        for i in range(len(conf_incorrect_matrix_cot.index)):
            for j in range(len(conf_incorrect_matrix_cot.columns)):
                text = axes[1,0].text(j, i, f'{conf_incorrect_matrix_cot.iloc[i, j]:.3f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im3, ax=axes[1,0], fraction=0.046, pad=0.04)
    
    # Plot 4: Mean confidence when incorrect by family and dataset (ZS)
    if len(zs_data) > 0:
        conf_incorrect_matrix_zs = zs_data.groupby(['family', 'dataset'])['mean_conf_incorrect'].mean().unstack(fill_value=0)
        im4 = axes[1,1].imshow(conf_incorrect_matrix_zs.values, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
        axes[1,1].set_xticks(range(len(conf_incorrect_matrix_zs.columns)))
        axes[1,1].set_xticklabels(conf_incorrect_matrix_zs.columns, rotation=45)
        axes[1,1].set_yticks(range(len(conf_incorrect_matrix_zs.index)))
        axes[1,1].set_yticklabels(conf_incorrect_matrix_zs.index)
        axes[1,1].set_title(f'{EXP_TYPE_MAPPING["zs_exp"]}\nMean Confidence When Incorrect', fontweight='bold')
        
        # Add text annotations
        for i in range(len(conf_incorrect_matrix_zs.index)):
            for j in range(len(conf_incorrect_matrix_zs.columns)):
                text = axes[1,1].text(j, i, f'{conf_incorrect_matrix_zs.iloc[i, j]:.3f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im4, ax=axes[1,1], fraction=0.046, pad=0.04)
    
    # Plot 5: Confidence gap comparison by family
    family_conf_gap = summary_df.groupby(['family', 'exp_type'])['confidence_gap'].mean().unstack()
    if 'cot_exp' in family_conf_gap.columns and 'zs_exp' in family_conf_gap.columns:
        x_pos = np.arange(len(families))
        width = 0.35
        
        bars1 = axes[0,2].bar(x_pos - width/2, family_conf_gap['cot_exp'], width, 
                       label=f'{EXP_TYPE_MAPPING["cot_exp"]}', alpha=0.8, 
                       color=[family_colors.get(f, '#888888') for f in families])
        bars2 = axes[0,2].bar(x_pos + width/2, family_conf_gap['zs_exp'], width,
                       label=f'{EXP_TYPE_MAPPING["zs_exp"]}', alpha=0.8,
                       color=[family_colors.get(f, '#888888') for f in families])
        
        axes[0,2].set_xlabel('Model Family')
        axes[0,2].set_ylabel('Mean Confidence Gap')
        axes[0,2].set_title('Confidence Gap by Family and Method', fontweight='bold')
        axes[0,2].set_xticks(x_pos)
        axes[0,2].set_xticklabels(families)
        axes[0,2].legend()
        apply_clean_style(axes[0,2])
    
    # Plot 6: Model count by family
    family_counts = summary_df.groupby('family')['model'].nunique()
    bars = axes[1,2].bar(family_counts.index, family_counts.values, 
                  color=[family_colors.get(f, '#888888') for f in family_counts.index],
                  alpha=0.8, edgecolor='black', linewidth=1)
    axes[1,2].set_xlabel('Model Family')
    axes[1,2].set_ylabel('Number of Models')
    axes[1,2].set_title(f'Model Count by Family\n(Quantized: {"Included" if quant else "Excluded"})', 
                 fontweight='bold')
    
    # Add value labels on bars
    for bar, count in zip(bars, family_counts.values):
        height = bar.get_height()
        axes[1,2].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    apply_clean_style(axes[1,2])
    
    plt.tight_layout()
    
    # Save overview plot
    fig.savefig(os.path.join(save_dir, 'family_confidence_overview.pdf'), 
               dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, 'family_confidence_overview.png'), 
               dpi=300, bbox_inches='tight')
    plt.close(fig)
    
def create_family_confidence_distributions(all_data: Dict, dataset: str, exp_type: str, 
                                         family_colors: Dict[str, str], save_dir: str):
    """Create confidence distribution plots grouped by model families for a specific dataset and experiment type"""
    
    # Collect all confidence data by family for this dataset/experiment
    family_data = {}
    
    for key, df_info in all_data.items():
        df, metadata = df_info
        if df is None:
            continue
            
        # Filter for this specific dataset and experiment type
        if metadata['dataset'] != dataset or metadata['exp_type'] != exp_type:
            continue
            
        family = categorize_model_family(metadata['model_name'])
        
        if family not in family_data:
            family_data[family] = {
                'correct': [],
                'incorrect': [],
                'models': []
            }
        
        # Collect confidence data
        correct_data = df[df['correct'] == True]['p_true'].values
        incorrect_data = df[df['correct'] == False]['p_true'].values
        
        family_data[family]['correct'].extend(correct_data)
        family_data[family]['incorrect'].extend(incorrect_data)
        family_data[family]['models'].append(metadata['model_name'])
    
    if not family_data:
        return
    
    # Determine experiment name for titles and filenames
    exp_name = EXP_TYPE_MAPPING[exp_type]
    exp_short = "cot" if exp_type == "cot_exp" else "zs"
    
    # Create separate plots for correct and incorrect confidence distributions
    bins = np.linspace(0, 1, 21)
    
    # Colors for correct vs incorrect (consistent across all plots)
    correct_color = '#2E8B57'    # Sea green
    incorrect_color = '#DC143C'  # Crimson
    
    # Plot 1: Confidence when Correct
    fig1, ax1 = plt.subplots(figsize=SUMMARY_PLOT_SIZE)
    
    for family in sorted(family_data.keys()):
        data = family_data[family]
        if len(data['correct']) == 0:
            continue
            
        color = family_colors.get(family, '#888888')
        mean_conf = np.mean(data['correct'])
        
        ax1.hist(data['correct'], bins=bins, alpha=0.7, density=True,
                label=f"{family} (μ={mean_conf:.3f}, n={len(data['models'])})", 
                color=color, edgecolor='white', linewidth=0.5)
    
    apply_clean_style(ax1)
    ax1.set_xlabel('Confidence (p_true)', fontweight='normal', fontsize=12)
    ax1.set_ylabel('Density', fontweight='normal', fontsize=12)
    ax1.set_title(f'Family Confidence Distributions - When Correct\n{dataset.upper()} - {exp_name}', 
                 fontweight='bold', pad=20, fontsize=14)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(bottom=0)
    ax1.set_xticks(np.arange(0, 1.1, 0.2))
    
    legend1 = ax1.legend(loc='upper left', frameon=True, fancybox=False,
                        edgecolor='#333333', facecolor='white', 
                        framealpha=0.95, fontsize=11, title='Model Families',
                        title_fontsize=12)
    legend1.get_frame().set_linewidth(1)
    
    plt.tight_layout()
    
    # Save correct confidence plot
    filename_correct = f'family_confidence_correct_{dataset.lower()}_{exp_short}'
    fig1.savefig(os.path.join(save_dir, f'{filename_correct}.pdf'), 
                           dpi=300, bbox_inches='tight')
    fig1.savefig(os.path.join(save_dir, f'{filename_correct}.png'), 
                           dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Plot 2: Confidence when Incorrect
    fig2, ax2 = plt.subplots(figsize=SUMMARY_PLOT_SIZE)
    
    for family in sorted(family_data.keys()):
        data = family_data[family]
        if len(data['incorrect']) == 0:
            continue
            
        color = family_colors.get(family, '#888888')
        mean_conf = np.mean(data['incorrect'])
        
        ax2.hist(data['incorrect'], bins=bins, alpha=0.7, density=True,
                label=f"{family} (μ={mean_conf:.3f}, n={len(data['models'])})", 
                color=color, edgecolor='white', linewidth=0.5)
    
    apply_clean_style(ax2)
    ax2.set_xlabel('Confidence (p_true)', fontweight='normal', fontsize=12)
    ax2.set_ylabel('Density', fontweight='normal', fontsize=12)
    ax2.set_title(f'Family Confidence Distributions - When Incorrect\n{dataset.upper()} - {exp_name}', 
                 fontweight='bold', pad=20, fontsize=14)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(bottom=0)
    ax2.set_xticks(np.arange(0, 1.1, 0.2))
    
    legend2 = ax2.legend(loc='upper right', frameon=True, fancybox=False,
                        edgecolor='#333333', facecolor='white', 
                        framealpha=0.95, fontsize=11, title='Model Families',
                        title_fontsize=12)
    legend2.get_frame().set_linewidth(1)
    
    plt.tight_layout()
    
    # Save incorrect confidence plot
    filename_incorrect = f'family_confidence_incorrect_{dataset.lower()}_{exp_short}'
    fig2.savefig(os.path.join(save_dir, f'{filename_incorrect}.pdf'), 
               dpi=300, bbox_inches='tight')
    fig2.savefig(os.path.join(save_dir, f'{filename_incorrect}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    # Plot 3: Combined plot (correct vs incorrect by family)
    fig3, ax3 = plt.subplots(figsize=LARGE_PLOT_SIZE)
    
    # Create subplots for each family
    n_families = len(family_data)
    if n_families == 0:
        return
        
    # Use a grid layout for multiple families
    if n_families <= 2:
        fig3, axes = plt.subplots(1, n_families, figsize=LARGE_PLOT_SIZE)
        if n_families == 1:
            axes = [axes]
    elif n_families <= 4:
        fig3, axes = plt.subplots(2, 2, figsize=LARGE_PLOT_SIZE)
        axes = axes.flatten()
    else:
        fig3, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
    
    for i, family in enumerate(sorted(family_data.keys())):
        if i >= len(axes):
            break
            
        data = family_data[family]
        ax = axes[i]
        
        if len(data['correct']) == 0 and len(data['incorrect']) == 0:
            continue
        
        # Plot correct and incorrect distributions with consistent colors
        if len(data['correct']) > 0:
            mean_correct = np.mean(data['correct'])
            ax.hist(data['correct'], bins=bins, alpha=0.7, density=True,
                   label=f"Correct (μ={mean_correct:.3f})", 
                   color=correct_color, edgecolor='white', linewidth=0.5)
        
        if len(data['incorrect']) > 0:
            mean_incorrect = np.mean(data['incorrect'])
            ax.hist(data['incorrect'], bins=bins, alpha=0.7, density=True,
                   label=f"Incorrect (μ={mean_incorrect:.3f})", 
                   color=incorrect_color, edgecolor='white', linewidth=0.5)
        
        apply_clean_style(ax)
        ax.set_xlabel('Confidence (p_true)', fontweight='normal')
        ax.set_ylabel('Density', fontweight='normal')
        ax.set_title(f'{family} Family\n(n={len(data["models"])} models)', 
                    fontweight='bold', pad=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        
        legend = ax.legend(loc='upper left', frameon=True, fancybox=False,
                          edgecolor='#cccccc', facecolor='white', framealpha=0.9)
        legend.get_frame().set_linewidth(0.5)
    
    # Hide unused subplots
    for i in range(len(family_data), len(axes)):
        axes[i].set_visible(False)
    
    fig3.suptitle(f'Family Confidence Distributions by Correctness\n{dataset.upper()} - {exp_name}', 
                 fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save combined plot
    filename_combined = f'family_confidence_combined_{dataset.lower()}_{exp_short}'
    fig3.savefig(os.path.join(save_dir, f'{filename_combined}.pdf'), 
               dpi=300, bbox_inches='tight')
    fig3.savefig(os.path.join(save_dir, f'{filename_combined}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    # Save family statistics
    family_stats = []
    for family in sorted(family_data.keys()):
        data = family_data[family]
        if len(data['correct']) > 0 or len(data['incorrect']) > 0:
            family_stats.append({
                'family': family,
                'dataset': dataset,
                'exp_type': exp_type,
                'n_models': len(data['models']),
                'models': ', '.join(data['models']),
                'mean_conf_correct': np.mean(data['correct']) if len(data['correct']) > 0 else np.nan,
                'std_conf_correct': np.std(data['correct']) if len(data['correct']) > 0 else np.nan,
                'mean_conf_incorrect': np.mean(data['incorrect']) if len(data['incorrect']) > 0 else np.nan,
                'std_conf_incorrect': np.std(data['incorrect']) if len(data['incorrect']) > 0 else np.nan,
                'n_correct_samples': len(data['correct']),
                'n_incorrect_samples': len(data['incorrect'])
            })
    
    if family_stats:
        family_stats_df = pd.DataFrame(family_stats)
        stats_filename = f'family_confidence_stats_{dataset.lower()}_{exp_short}.csv'
        family_stats_df.to_csv(os.path.join(save_dir, stats_filename), index=False)
    
    return filename_correct, filename_incorrect, filename_combined

def plots_for_single_model(all_data: Dict, model_name: str, save_dir: str = "single_model_analysis", quant: bool = False):
    """Create focused confidence distribution plots for a single model across all tasks/datasets
    
    Args:
        all_data: Dictionary containing processed data
        model_name: Name of the specific model to analyze
        save_dir: Directory to save single model analysis plots
        quant: Whether quantized models are included in the analysis
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Filter data for the specific model
    model_data = {}
    for key, df_info in all_data.items():
        df, metadata = df_info
        if df is not None and metadata['model_name'] == model_name:
            dataset = metadata['dataset']
            exp_type = metadata['exp_type']
            
            if dataset not in model_data:
                model_data[dataset] = {}
            
            model_data[dataset][exp_type] = {
                'df': df,
                'metadata': metadata
            }
    
    if not model_data:
        print(f"❌ No data found for model: {model_name}")
        return
    
    datasets = sorted(model_data.keys())
    
    print(f"🤖 Creating single model confidence distribution analysis for: {model_name}")
    print(f"   Datasets: {', '.join(datasets)}")
    print(f"   Quantized models: {'Included' if quant else 'Excluded'}")
    
    # Create confidence distribution plots for each dataset
    created_plots = []
    
    for dataset in datasets:
        dataset_info = model_data[dataset]
        exp_types = sorted(dataset_info.keys())
        
        print(f"   📊 Processing {dataset.upper()} - {', '.join(['CoT' if exp == 'cot_exp' else 'ZS' for exp in exp_types])}...")
        
        # Create confidence distribution plots for this dataset
        result = create_single_model_confidence_distributions(
            dataset_info, model_name, dataset, save_dir
        )
        
        if result:
            created_plots.extend(result)
    
    # Create overall summary statistics for this model
    create_single_model_summary_statistics(all_data, model_name, save_dir, quant)
    
    print(f"📊 Single model confidence distribution analysis complete!")
    print(f"   Results saved in '{save_dir}/' directory:")
    
    # Group and display created plots
    for dataset in datasets:
        print(f"   • {dataset.upper()}:")
        print(f"     - single_model_confidence_{dataset.lower()}.pdf/png")
        print(f"     - single_model_stats_{dataset.lower()}.csv")
    
    print(f"   • Overall Analysis:")
    print(f"     - single_model_summary_statistics.csv")
    print(f"     - single_model_overview.pdf/png")

def create_single_model_confidence_distributions(dataset_info: Dict, model_name: str, 
                                               dataset: str, save_dir: str):
    """Create confidence distribution plots for a single model across experiment types for one dataset"""
    
    # Sort experiment types to put ZS first, then CoT
    exp_types = sorted(dataset_info.keys(), key=lambda x: 0 if x == 'zs_exp' else 1)
    
    if len(exp_types) == 0:
        return []
    
    # Colors for correct vs incorrect (consistent across all plots)
    correct_color = '#2E8B57'    # Sea green
    incorrect_color = '#DC143C'  # Crimson
    
    # Colors for experiment type borders/accents
    exp_border_colors = {'cot_exp': '#1B5E20', 'zs_exp': '#B71C1C'}  # Darker versions for borders
    
    # Create combined plot showing both correct and incorrect for all experiment types
    fig, axes = plt.subplots(2, len(exp_types), figsize=(6 * len(exp_types), 10))
    
    # Handle case where there's only one experiment type
    if len(exp_types) == 1:
        axes = axes.reshape(2, 1)
    
    bins = np.linspace(0, 1, 21)
    
    # Clean model name for display using mapping
    model_display = MODEL_NAME_MAPPING.get(model_name, model_name.replace('-', ' ').replace('_', ' '))
    
    for i, exp_type in enumerate(exp_types):
        data_info = dataset_info[exp_type]
        df = data_info['df']
        
        correct_data = df[df['correct'] == True]['p_true'].values
        incorrect_data = df[df['correct'] == False]['p_true'].values
        
        exp_name = EXP_TYPE_MAPPING[exp_type]
        border_color = exp_border_colors[exp_type]
        
        # Top subplot: Confidence when correct
        ax_correct = axes[0, i]
        if len(correct_data) > 0:
            mean_correct = np.mean(correct_data)
            ax_correct.hist(correct_data, bins=bins, alpha=0.7, density=True,
                           color=correct_color,
                           label=f"μ={mean_correct:.3f}")
        
        apply_clean_style(ax_correct)
        ax_correct.set_xlabel('Confidence (p_true)', fontweight='normal')
        ax_correct.set_ylabel('Density', fontweight='normal')
        ax_correct.set_title(f'{exp_name}\nWhen Correct', fontweight='bold', pad=10)
        ax_correct.set_xlim(0, 1)
        ax_correct.set_ylim(bottom=0)
        ax_correct.set_xticks(np.arange(0, 1.1, 0.2))
        
        if len(correct_data) > 0:
            legend = ax_correct.legend(loc='upper left', frameon=True, fancybox=False,
                                     edgecolor='#cccccc', facecolor='white', framealpha=0.9)
            legend.get_frame().set_linewidth(0.5)
        
        # Bottom subplot: Confidence when incorrect
        ax_incorrect = axes[1, i]
        if len(incorrect_data) > 0:
            mean_incorrect = np.mean(incorrect_data)
            ax_incorrect.hist(incorrect_data, bins=bins, alpha=0.7, density=True,
                             color=incorrect_color,
                             label=f"μ={mean_incorrect:.3f}")
        
        apply_clean_style(ax_incorrect)
        ax_incorrect.set_xlabel('Confidence (p_true)', fontweight='normal')
        ax_incorrect.set_ylabel('Density', fontweight='normal')
        ax_incorrect.set_title(f'{exp_name}\nWhen Incorrect', fontweight='bold', pad=10)
        ax_incorrect.set_xlim(0, 1)
        ax_incorrect.set_ylim(bottom=0)
        ax_incorrect.set_xticks(np.arange(0, 1.1, 0.2))
        
        if len(incorrect_data) > 0:
            legend = ax_incorrect.legend(loc='upper right', frameon=True, fancybox=False,
                                       edgecolor='#cccccc', facecolor='white', framealpha=0.9)
            legend.get_frame().set_linewidth(0.5)
    
    fig.suptitle(f'Model Confidence Distributions\n{model_display} - {dataset.upper()}', 
                 fontweight='bold', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save the plot
    filename = f'{model_name}_{dataset.lower()}'
    fig.savefig(os.path.join(save_dir, f'{filename}.pdf'), 
               dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, f'{filename}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save statistics for this dataset
    stats_data = []
    for exp_type in exp_types:
        data_info = dataset_info[exp_type]
        df = data_info['df']
        
        correct_data = df[df['correct'] == True]['p_true'].values
        incorrect_data = df[df['correct'] == False]['p_true'].values
        
        stats_data.append({
            'model': model_name,
            'dataset': dataset,
            'exp_type': exp_type,
            'accuracy': (df['correct'] == True).mean(),
            'mean_conf_correct': np.mean(correct_data) if len(correct_data) > 0 else np.nan,
            'std_conf_correct': np.std(correct_data) if len(correct_data) > 0 else np.nan,
            'mean_conf_incorrect': np.mean(incorrect_data) if len(incorrect_data) > 0 else np.nan,
            'std_conf_incorrect': np.std(incorrect_data) if len(incorrect_data) > 0 else np.nan,
            'confidence_gap': (np.mean(correct_data) - np.mean(incorrect_data)) if len(correct_data) > 0 and len(incorrect_data) > 0 else np.nan,
            'n_correct_samples': len(correct_data),
            'n_incorrect_samples': len(incorrect_data),
            'total_samples': len(df)
        })
    
    if stats_data:
        stats_df = pd.DataFrame(stats_data)
        stats_filename = f'single_model_stats_{dataset.lower()}.csv'
        stats_df.to_csv(os.path.join(save_dir, stats_filename), index=False)
    
    return [filename]

def create_single_model_summary_statistics(all_data: Dict, model_name: str, save_dir: str, quant: bool):
    """Create overall summary statistics and overview plot for single model analysis"""
    
    # Aggregate data for this specific model
    summary_data = []
    
    for key, df_info in all_data.items():
        df, metadata = df_info
        if df is None or metadata['model_name'] != model_name:
            continue
            
        correct_data = df[df['correct'] == True]['p_true']
        incorrect_data = df[df['correct'] == False]['p_true']
        
        summary_data.append({
            'model': metadata['model_name'],
            'dataset': metadata['dataset'],
            'split': metadata['split'],
            'exp_type': metadata['exp_type'],
            'accuracy': (df['correct'] == True).mean(),
            'mean_conf_correct': correct_data.mean(),
            'mean_conf_incorrect': incorrect_data.mean(),
            'confidence_gap': correct_data.mean() - incorrect_data.mean(),
            'n_samples': len(df)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    if len(summary_df) == 0:
        print(f"❌ No data available for model: {model_name}")
        return
    
    # Save comprehensive statistics
    summary_df.to_csv(os.path.join(save_dir, 'single_model_summary_statistics.csv'), index=False)
    
    # Create overview plot showing model performance across datasets and experiment types
    datasets = sorted(summary_df['dataset'].unique())
    exp_types = sorted(summary_df['exp_type'].unique())
    
    # Clean model name for display
    model_display = model_name.replace('-', ' ').replace('_', ' ')
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Model Performance Overview: {model_display}', fontsize=16, fontweight='bold')
    
    # Colors for experiment types
    exp_colors = {'cot_exp': '#2E8B57', 'zs_exp': '#DC143C'}
    exp_names = {'cot_exp': 'CoT', 'zs_exp': 'ZS'}
    
    # Plot 1: Accuracy by dataset and experiment type
    x_pos = np.arange(len(datasets))
    width = 0.35
    
    for i, exp_type in enumerate(exp_types):
        exp_data = summary_df[summary_df['exp_type'] == exp_type]
        accuracies = []
        
        for dataset in datasets:
            dataset_data = exp_data[exp_data['dataset'] == dataset]
            if len(dataset_data) > 0:
                accuracies.append(dataset_data['accuracy'].iloc[0])
            else:
                accuracies.append(0)
        
        bars = axes[0,0].bar(x_pos + i * width, accuracies, width, 
                           label=exp_names[exp_type], alpha=0.8,
                           color=exp_colors[exp_type])
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            if height > 0:
                axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    axes[0,0].set_xlabel('Dataset')
    axes[0,0].set_ylabel('Accuracy')
    axes[0,0].set_title('Accuracy by Dataset and Experiment Type', fontweight='bold')
    axes[0,0].set_xticks(x_pos + width/2)
    axes[0,0].set_xticklabels([d.upper() for d in datasets])
    axes[0,0].legend()
    axes[0,0].set_ylim(0, 1)
    apply_clean_style(axes[0,0])
    
    # Plot 2: Confidence gap by dataset and experiment type
    for i, exp_type in enumerate(exp_types):
        exp_data = summary_df[summary_df['exp_type'] == exp_type]
        conf_gaps = []
        
        for dataset in datasets:
            dataset_data = exp_data[exp_data['dataset'] == dataset]
            if len(dataset_data) > 0:
                conf_gaps.append(dataset_data['confidence_gap'].iloc[0])
            else:
                conf_gaps.append(0)
        
        bars = axes[0,1].bar(x_pos + i * width, conf_gaps, width,
                           label=exp_names[exp_type], alpha=0.8,
                           color=exp_colors[exp_type])
        
        # Add value labels on bars
        for bar, gap in zip(bars, conf_gaps):
            height = bar.get_height()
            if abs(height) > 0.001:
                axes[0,1].text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.01),
                              f'{gap:.3f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    axes[0,1].set_xlabel('Dataset')
    axes[0,1].set_ylabel('Confidence Gap (Correct - Incorrect)')
    axes[0,1].set_title('Confidence Gap by Dataset and Experiment Type', fontweight='bold')
    axes[0,1].set_xticks(x_pos + width/2)
    axes[0,1].set_xticklabels([d.upper() for d in datasets])
    axes[0,1].legend()
    axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    apply_clean_style(axes[0,1])
    
    # Plot 3: Mean confidence when correct
    for i, exp_type in enumerate(exp_types):
        exp_data = summary_df[summary_df['exp_type'] == exp_type]
        conf_correct = []
        
        for dataset in datasets:
            dataset_data = exp_data[exp_data['dataset'] == dataset]
            if len(dataset_data) > 0:
                conf_correct.append(dataset_data['mean_conf_correct'].iloc[0])
            else:
                conf_correct.append(0)
        
        bars = axes[1,0].bar(x_pos + i * width, conf_correct, width,
                           label=exp_names[exp_type], alpha=0.8,
                           color='#2E8B57')  # Green for correct
        
        # Add value labels on bars
        for bar, conf in zip(bars, conf_correct):
            height = bar.get_height()
            if height > 0:
                axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'{conf:.3f}', ha='center', va='bottom', fontsize=9)
    
    axes[1,0].set_xlabel('Dataset')
    axes[1,0].set_ylabel('Mean Confidence When Correct')
    axes[1,0].set_title('Confidence When Correct by Dataset and Experiment Type', fontweight='bold')
    axes[1,0].set_xticks(x_pos + width/2)
    axes[1,0].set_xticklabels([d.upper() for d in datasets])
    axes[1,0].legend()
    axes[1,0].set_ylim(0, 1)
    apply_clean_style(axes[1,0])
    
    # Plot 4: Mean confidence when incorrect
    for i, exp_type in enumerate(exp_types):
        exp_data = summary_df[summary_df['exp_type'] == exp_type]
        conf_incorrect = []
        
        for dataset in datasets:
            dataset_data = exp_data[exp_data['dataset'] == dataset]
            if len(dataset_data) > 0:
                conf_incorrect.append(dataset_data['mean_conf_incorrect'].iloc[0])
            else:
                conf_incorrect.append(0)
        
        bars = axes[1,1].bar(x_pos + i * width, conf_incorrect, width,
                           label=exp_names[exp_type], alpha=0.8,
                           color='#DC143C')  # Red for incorrect
        
        # Add value labels on bars
        for bar, conf in zip(bars, conf_incorrect):
            height = bar.get_height()
            if height > 0:
                axes[1,1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'{conf:.3f}', ha='center', va='bottom', fontsize=9)
    
    axes[1,1].set_xlabel('Dataset')
    axes[1,1].set_ylabel('Mean Confidence When Incorrect')
    axes[1,1].set_title('Confidence When Incorrect by Dataset and Experiment Type', fontweight='bold')
    axes[1,1].set_xticks(x_pos + width/2)
    axes[1,1].set_xticklabels([d.upper() for d in datasets])
    axes[1,1].legend()
    axes[1,1].set_ylim(0, 1)
    apply_clean_style(axes[1,1])
    
    plt.tight_layout()
    
    # Save overview plot
    fig.savefig(os.path.join(save_dir, 'single_model_overview.pdf'), 
               dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, 'single_model_overview.png'), 
               dpi=300, bbox_inches='tight')
    plt.close(fig)

if __name__ == "__main__":
    import sys
    
    # Check for help
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit(0)
    
    # Check for flags
    summary_only = "--summary-only" in sys.argv or "-s" in sys.argv
    quant = "--quant" in sys.argv or "-q" in sys.argv
    family_only = "--family-only" in sys.argv or "-f" in sys.argv
    no_sample = "--no-sample" in sys.argv
    list_models = "--list-models" in sys.argv or "-l" in sys.argv
    
    # Check for model name
    model_name = None
    for i, arg in enumerate(sys.argv):
        if arg == "--model" or arg == "-m":
            if i + 1 < len(sys.argv):
                model_name = sys.argv[i + 1]
                break
    
    # Handle list models request
    if list_models:
        print("📋 Discovering available models...")
        base_dirs = ['cot_exp', 'zs_exp']
        available_models = list_available_models(base_dirs, quant=quant)
        
        if not available_models:
            print("❌ No models found in the data files!")
        else:
            print(f"✅ Found {len(available_models)} unique models:")
            print(f"   Quantized models: {'Included' if quant else 'Excluded'}")
            print()
            
            # Group by family for better organization
            family_groups = {}
            for model in available_models:
                family = categorize_model_family(model)
                if family not in family_groups:
                    family_groups[family] = []
                family_groups[family].append(model)
            
            for family in sorted(family_groups.keys()):
                print(f"🏠 {family} Family:")
                for model in sorted(family_groups[family]):
                    print(f"   • {model}")
                print()
            
            print("💡 Use --model <model_name> to analyze a specific model")
            print("💡 Use --quant to include/exclude quantized models")
        
        sys.exit(0)
    
    # Validate flag combinations
    exclusive_flags = [summary_only, family_only, bool(model_name)]
    if sum(exclusive_flags) > 1:
        print("❌ Cannot use multiple exclusive flags together:")
        print("   --summary-only (-s): Generate only summary plots")
        print("   --family-only (-f): Generate only family analysis plots")
        print("   --model (-m): Generate plots for a single model")
        print("   --list-models (-l): List all available models")
        sys.exit(1)
    
    if model_name:
        print(f"🎯 Running in single model mode for: {model_name}")
    elif family_only:
        print("🎯 Running in family-only mode (family analysis plots only)")
    elif summary_only:
        print("🎯 Running in summary-only mode (no individual confidence histograms)")
    
    if no_sample:
        print("🎯 Running with --no-sample flag (loading all data)")
    
    main(summary_only=summary_only, quant=quant, family_only=family_only, 
         sample=not no_sample, model_name=model_name) 