import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
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

# Import common utilities
from plotting_utils import (
    MODEL_NAME_MAPPING, EXP_TYPE_MAPPING, INDIVIDUAL_PLOT_SIZE,
    setup_publication_style, apply_clean_style,
    categorize_model_family, discover_data_files, parse_filename,
    load_json_data, process_data_only, list_available_models
)

# Import specialized plot modules
from summary_plots import create_summary_plots


def create_confidence_histogram(df: pd.DataFrame, metadata: dict, 
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
        exp_type_mapping = {
            'cot_exp': 'cot',
            'zs_exp': 'zs',
            'verbalized': 'verbalized',
            'verbalized_cot': 'verbalized_cot',
            'otherAI': 'otherAI',
            'otherAI_cot': 'otherAI_cot'
        }
        exp_prefix = exp_type_mapping.get(metadata['exp_type'], metadata['exp_type']) + "_"
        filename = f"{exp_prefix}confidence_histogram_{metadata['dataset']}_{metadata['split']}_{metadata['model_name']}.pdf"
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        
        # Also save PNG for viewing
        png_filepath = filepath.replace('.pdf', '.png')
        fig.savefig(png_filepath, dpi=300, bbox_inches='tight')
    
    return fig


def process_single_file(filepath: str) -> tuple:
    """Process a single data file and generate plots"""
    try:
        metadata = parse_filename(filepath)
        
        # Skip train files
        if metadata['split'] == 'train':
            return filepath, None, False
        
        df = load_json_data(filepath, sample=False)
        if df is None:
            return filepath, None, False
        
        # Create output directories for density and raw plots
        exp_type_mapping = {
            'cot_exp': 'cot',
            'zs_exp': 'zs',
            'verbalized': 'verbalized',
            'verbalized_cot': 'verbalized_cot',
            'otherAI': 'otherAI',
            'otherAI_cot': 'otherAI_cot'
        }
        exp_short = exp_type_mapping.get(metadata['exp_type'], metadata['exp_type'])
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
    print("  -q, --quant          Include GGUF quantized models (excluded by default)")
    print()
    print("EXAMPLES:")
    print("  # Generate all plots (default)")
    print("  python generate_all_plots.py")
    print()
    print("  # Include quantized models")
    print("  python generate_all_plots.py --quant")
    print()
    print("OUTPUT DIRECTORIES:")
    print("  📂 plots/                    Individual confidence histograms")
    print()
    print("💡 All plots are saved in both PDF (vector) and PNG (raster) formats")
    print()
    print("SUPPORTED EXPERIMENT TYPES:")
    print("  • cot_exp                    Chain-of-thought experiments")
    print("  • zs_exp                     Zero-shot experiments")
    print("  • verbalized                 Verbalized confidence experiments")
    print("  • verbalized_cot             Verbalized confidence with CoT")
    print("  • otherAI                    Other AI evaluation experiments")
    print("  • otherAI_cot                Other AI evaluation with CoT")


def main(quant: bool = False):
    """Main function to generate confidence histogram plots
    
    Args:
        quant: If False (default), exclude GGUF quantized models. If True, include them.
    """
    
    # Setup publication style
    setup_publication_style()
    
    if not quant:
        print("🚫 Excluding GGUF quantized models (use --quant to include them)")
    else:
        print("✅ Including GGUF quantized models")
    
    # Discover all data files
    base_dirs = ['cot_exp', 'zs_exp', 'verbalized', 'verbalized_cot', 'otherAI', 'otherAI_cot']
    data_files = discover_data_files(base_dirs, quant=quant)
    
    if not data_files:
        print("❌ No data files found!")
        return
    
    # Collect all file paths and filter out archive folders
    all_files = []
    for base_dir, files in data_files.items():
        # Filter out files in archive or tracing folders
        filtered_files = [f for f in files if "archive" not in f.lower() and "tracing" not in f.lower()]
        all_files.extend(filtered_files)
    
    print(f"📊 Found {len(all_files)} data files across {len(data_files)} experiment types (excluding archive and tracing folders)")
    
    # Process all files with ThreadPoolExecutor
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_single_file, filepath): filepath 
                         for filepath in all_files}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(all_files), desc="Processing files", unit="file") as pbar:
            for future in as_completed(future_to_file):
                filepath = future_to_file[future]
                try:
                    result_filepath, data_result, success = future.result()
                    
                    if success and data_result is not None:
                        pbar.set_postfix_str(f"✅ {os.path.basename(filepath)}")
                    else:
                        failed_files.append(result_filepath)
                        pbar.set_postfix_str(f"❌ {os.path.basename(filepath)}")
                        
                except Exception as e:
                    failed_files.append(filepath)
                    pbar.set_postfix_str(f"❌ {os.path.basename(filepath)} - {str(e)}")
                
                pbar.update(1)
    
    # Print summary
    successful_files = len(all_files) - len(failed_files)
    print(f"\n✅ Processing complete!")
    print(f"   Successfully processed: {successful_files} files")
    print(f"   Failed files: {len(failed_files)}")
    
    if failed_files:
        print("   Failed files:")
        for f in failed_files:
            print(f"     {f}")
    
    print(f"\n📁 Plots saved in organized structure:")
    print(f"   📂 plots/cot/density/{'{dataset}'}/")
    print(f"   📂 plots/cot/raw/{'{dataset}'}/")
    print(f"   📂 plots/zs/density/{'{dataset}'}/")
    print(f"   📂 plots/zs/raw/{'{dataset}'}/")
    print(f"   📂 plots/verbalized/density/{'{dataset}'}/")
    print(f"   📂 plots/verbalized/raw/{'{dataset}'}/")
    print(f"   📂 plots/verbalized_cot/density/{'{dataset}'}/")
    print(f"   📂 plots/verbalized_cot/raw/{'{dataset}'}/")
    print(f"   📂 plots/otherAI/density/{'{dataset}'}/")
    print(f"   📂 plots/otherAI/raw/{'{dataset}'}/")
    print(f"   📂 plots/otherAI_cot/density/{'{dataset}'}/")
    print(f"   📂 plots/otherAI_cot/raw/{'{dataset}'}/")
    
    print(f"💡 All plots are saved in both PDF (vector) and PNG (raster) formats")


if __name__ == "__main__":
    import sys
    
    # Check for help
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit(0)
    
    # Check for flags
    quant = "--quant" in sys.argv or "-q" in sys.argv
    
    main(quant=quant)