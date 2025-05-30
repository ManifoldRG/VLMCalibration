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
from family_plots import plots_across_families
from single_model_plots import plots_for_single_model


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
        exp_prefix = "cot_" if metadata['exp_type'] == 'cot_exp' else "zs_"
        filename = f"{exp_prefix}confidence_histogram_{metadata['dataset']}_{metadata['split']}_{metadata['model_name']}.pdf"
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        
        # Also save PNG for viewing
        png_filepath = filepath.replace('.pdf', '.png')
        fig.savefig(png_filepath, dpi=300, bbox_inches='tight')
    
    return fig


def process_single_file(filepath: str, sample: bool = True) -> tuple:
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