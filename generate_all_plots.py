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

def load_json_data(filepath: str) -> pd.DataFrame:
    """Load and process JSON data file"""
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
    bins = np.linspace(0, 1, 51)
    
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
    
    # Create a clean title
    model_display = metadata['model_name'].replace('-', ' ').replace('_', ' ')
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
        ax.set_aspect('equal', adjustable='box')
        
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
        ax2.set_aspect('equal', adjustable='box')
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
    ax1.set_title('Model Performance: CoT vs Zero-Shot\nAccuracy Improvement', fontweight='normal', pad=15)
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
    ax2.set_title('Model Performance: CoT vs Zero-Shot\nConfidence Gap Improvement', fontweight='normal', pad=15)
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
        exp_name = "Chain-of-Thought (CoT)" if exp_type == "cot_exp" else "Zero-Shot (ZS)"
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
        exp_name = "Chain-of-Thought" if exp_type == "cot_exp" else "Zero-Shot"
        print(f"   • {exp_name} ({exp_short.upper()}):")
        print(f"     - {exp_short}_calibration_overview.pdf/png")
        print(f"     - {exp_short}_calibration_overview_annotated.pdf/png")
        print(f"     - summary_statistics_{exp_short}.csv")
    print(f"   • CoT vs ZS Comparison:")
    print(f"     - cot_vs_zs_comparison.pdf/png")
    print(f"     - cot_vs_zs_comparison_data.csv")
    print(f"   • Combined: summary_statistics.csv")

def process_data_only(filepath: str) -> Tuple[str, tuple, bool]:
    """Process a single data file for summary statistics without generating plots"""
    try:
        metadata = parse_filename(filepath)
        
        df = load_json_data(filepath)
        if df is None:
            return filepath, None, False
        
        # Create key for all_data dictionary
        key = f"{metadata['exp_type']}_{metadata['dataset']}_{metadata['split']}_{metadata['model_name']}"
        
        return filepath, (key, (df, metadata)), True
        
    except Exception as e:
        return filepath, None, False

def process_single_file(filepath: str) -> Tuple[str, tuple, bool]:
    """Process a single data file and generate plots"""
    try:
        metadata = parse_filename(filepath)
        
        df = load_json_data(filepath)
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

def main(summary_only: bool = False):
    """Main function to generate all plots
    
    Args:
        summary_only: If True, only generate summary plots without individual confidence histograms
    """
    
    # Setup publication style
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
    
    # Choose processing function based on mode
    process_func = process_data_only if summary_only else process_single_file
    mode_description = "Processing data for summary" if summary_only else "Processing files"
    
    # Process all files with ThreadPoolExecutor
    all_data = {}
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_func, filepath): filepath 
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
    
    if summary_only:
        print(f"\n📁 Summary plots saved in 'summary_plots/' directory")
    else:
        print(f"\n📁 Plots saved in organized structure:")
        print(f"   📂 plots/cot/density/{'{dataset}'}/")
        print(f"   📂 plots/cot/raw/{'{dataset}'}/")
        print(f"   📂 plots/zs/density/{'{dataset}'}/")
        print(f"   📂 plots/zs/raw/{'{dataset}'}/")
        print(f"📁 Summary plots saved in 'summary_plots/' directory")
    
    print(f"💡 All plots are saved in both PDF (vector) and PNG (raster) formats")

if __name__ == "__main__":
    import sys
    
    # Check for summary-only flag
    summary_only = "--summary-only" in sys.argv or "-s" in sys.argv
    
    if summary_only:
        print("🎯 Running in summary-only mode (no individual confidence histograms)")
    
    main(summary_only=summary_only) 