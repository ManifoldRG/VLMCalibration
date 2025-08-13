"""Common utilities for confidence calibration plotting"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import json
import glob
import os
from typing import Dict, List, Tuple

# Model name mapping for cleaner display
MODEL_NAME_MAPPING = {
    'Llama-32-1B-Instruct': 'Llama 3.2 1B Instruct',
    'Llama-32-3B-Instruct': 'Llama 3.2 3B Instruct',
    'Llama-31-8B-Instruct': 'Llama 3.1 8B Instruct',
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
    'verbalized': 'Verbalized Confidence',
    'verbalized_cot': 'Verbalized Confidence + CoT',
    'otherAI': 'Other AI Evaluation',
    'otherAI_cot': 'Other AI Evaluation + CoT',
    'otherAI_verbalized': 'Other AI Verbalized Confidence',
    'otherAI_verbalized_cot': 'Other AI Verbalized Confidence + CoT',
}

# Standardized plot sizes
INDIVIDUAL_PLOT_SIZE = (8, 6)  # Standardized size for per-model-dataset-split plots
SUMMARY_PLOT_SIZE = (12, 8)    # Size for summary/comparison plots
LARGE_PLOT_SIZE = (16, 10)     # Size for complex comparison plots

# Color palette for families
FAMILY_COLORS = {
    'Llama': '#1f77b4',    # Blue
    'Gemma': '#ff7f0e',    # Orange  
    'Qwen': '#2ca02c',     # Green
    'OLMo': '#d62728',     # Red
    'Other': '#9467bd'     # Purple
}

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
        
        # Filter out files in archive or tracing folders
        files = [f for f in files if "archive" not in f.lower() and "tracing" not in f.lower()]
        
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
    
    # Handle different experiment type patterns
    if len(parts) >= 2 and parts[0] in ['cot', 'zs'] and parts[1] == 'exp':
        # Traditional format: cot_exp or zs_exp
        exp_type = f"{parts[0]}_{parts[1]}"
        records_idx = 2
    elif len(parts) >= 3 and parts[0] == 'verbalized' and parts[1] == 'cot':
        # verbalized_cot format
        exp_type = "verbalized_cot"
        records_idx = 2
    elif len(parts) >= 3 and parts[0] == 'otherAI' and parts[1] == 'cot':
        # otherAI_cot format
        exp_type = "otherAI_cot"
        records_idx = 2
    elif len(parts) >= 4 and parts[0] == 'otherAI' and parts[1] == 'verbalized' and parts[2] == 'cot':
        # otherAI_verbalized_cot format
        exp_type = "otherAI_verbalized_cot"
        records_idx = 3
    elif len(parts) >= 3 and parts[0] == 'otherAI' and parts[1] == 'verbalized':
        # otherAI_verbalized format
        exp_type = "otherAI_verbalized"
        records_idx = 2
    elif parts[0] == 'verbalized':
        # verbalized format
        exp_type = "verbalized"
        records_idx = 1
    elif parts[0] == 'otherAI':
        # otherAI format
        exp_type = "otherAI"
        records_idx = 1
    else:
        # Fallback to original logic for backward compatibility
        exp_type = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else parts[0]
        records_idx = 2
    
    # Find 'records' after the experiment type
    actual_records_idx = None
    for i in range(records_idx, len(parts)):
        if parts[i] == 'records':
            actual_records_idx = i
            break
    
    if actual_records_idx is None:
        # Fallback: assume records is at expected position
        actual_records_idx = records_idx
        split = parts[actual_records_idx + 1] if actual_records_idx + 1 < len(parts) else 'unknown'
        model_name = '_'.join(parts[actual_records_idx + 2:]) if actual_records_idx + 2 < len(parts) else 'unknown'
    else:
        split = parts[actual_records_idx + 1]
        model_name = '_'.join(parts[actual_records_idx + 2:])
    
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

def create_summary_dataframe(all_data: Dict) -> pd.DataFrame:
    """Create a summary DataFrame from all loaded data"""
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
    
    return pd.DataFrame(summary_data) 