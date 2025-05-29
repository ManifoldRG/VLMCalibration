#!/usr/bin/env python3
"""
Curate Top 200 Examples with Largest Confidence Shifts Between Zero-Shot and CoT

This script identifies examples where a model's confidence changed the most between
zero-shot and chain-of-thought prompting, helping to understand which types of 
problems benefit most from CoT reasoning.

Usage:
    python curate_confidence_shifts.py --model "gemma-2-9b-it" --output shifts_analysis.json
    python curate_confidence_shifts.py --model "Llama-32-1B-Instruct" --dataset gsm8k --top-k 100
"""

import pandas as pd
import numpy as np
import json
import os
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def discover_model_files(model_name: str, base_dirs: List[str] = None) -> Dict[str, Dict[str, str]]:
    """Discover all JSON files for a specific model across datasets and experiment types
    
    Args:
        model_name: Name of the model to search for
        base_dirs: List of base directories to search (default: ['cot_exp', 'zs_exp'])
    
    Returns:
        Dictionary with structure: {dataset: {exp_type: filepath}}
    """
    if base_dirs is None:
        base_dirs = ['cot_exp', 'zs_exp']
    
    model_files = {}
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
            
        # Find all JSON files that match the pattern for this model
        pattern = os.path.join(base_dir, "*", f"*_records_*_{model_name}.json")
        files = glob.glob(pattern)
        
        # Filter out experiment_details files and GGUF files
        files = [f for f in files if 'experiment_details' not in os.path.basename(f)]
        files = [f for f in files if 'gguf' not in os.path.basename(f).lower()]
        
        for filepath in files:
            # Extract dataset and experiment type
            dataset = os.path.basename(os.path.dirname(filepath))
            filename = os.path.basename(filepath)
            
            # Parse experiment type from filename
            if filename.startswith('cot_exp'):
                exp_type = 'cot'
            elif filename.startswith('zs_exp'):
                exp_type = 'zs'
            else:
                continue
            
            if dataset not in model_files:
                model_files[dataset] = {}
            
            model_files[dataset][exp_type] = filepath
    
    return model_files


def load_json_data(filepath: str) -> Optional[pd.DataFrame]:
    """Load and process JSON data file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # Check required columns
        required_cols = ['idx', 'p_true', 'correct', 'answer', 'true_answer']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols} in {filepath}")
            return None
        
        # Ensure idx is treated as string for consistent matching
        df['idx'] = df['idx'].astype(str)
        
        return df
        
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None


def calculate_confidence_shifts(zs_df: pd.DataFrame, cot_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate confidence shifts between zero-shot and CoT for matching examples
    
    Args:
        zs_df: Zero-shot experiment DataFrame
        cot_df: Chain-of-thought experiment DataFrame
    
    Returns:
        DataFrame with confidence shift analysis
    """
    # Merge on idx to find matching examples
    merged = pd.merge(zs_df, cot_df, on='idx', suffixes=('_zs', '_cot'))
    
    if len(merged) == 0:
        print("Warning: No matching examples found between ZS and CoT datasets")
        return pd.DataFrame()
    
    # Calculate confidence shifts
    merged['confidence_shift'] = merged['p_true_cot'] - merged['p_true_zs']
    merged['confidence_shift_abs'] = np.abs(merged['confidence_shift'])
    
    # Calculate accuracy changes
    merged['accuracy_change'] = merged['correct_cot'].astype(int) - merged['correct_zs'].astype(int)
    
    # Categorize the type of shift
    def categorize_shift(row):
        if row['correct_zs'] and row['correct_cot']:
            return 'correct_to_correct'
        elif not row['correct_zs'] and not row['correct_cot']:
            return 'incorrect_to_incorrect'
        elif not row['correct_zs'] and row['correct_cot']:
            return 'incorrect_to_correct'
        else:  # row['correct_zs'] and not row['correct_cot']
            return 'correct_to_incorrect'
    
    merged['shift_category'] = merged.apply(categorize_shift, axis=1)
    
    # Add additional analysis columns
    merged['zs_confidence_level'] = pd.cut(merged['p_true_zs'], 
                                          bins=[0, 0.3, 0.7, 1.0], 
                                          labels=['low', 'medium', 'high'])
    merged['cot_confidence_level'] = pd.cut(merged['p_true_cot'], 
                                           bins=[0, 0.3, 0.7, 1.0], 
                                           labels=['low', 'medium', 'high'])
    
    # Select and rename columns for output - include all relevant fields
    result_cols = [
        'idx',
        'question_zs',  # The problem statement
        'response_zs', 'response_cot',  # Model responses
        'answer_zs', 'answer_cot',  # Extracted answers
        'true_answer_zs',  # Ground truth (should be same for both)
        'p_true_zs', 'p_true_cot',  # Confidence scores
        'correct_zs', 'correct_cot',  # Correctness
        'confidence_shift', 'confidence_shift_abs',  # Shift metrics
        'accuracy_change', 'shift_category',  # Analysis
        'zs_confidence_level', 'cot_confidence_level'  # Confidence levels
    ]
    
    # Add inject_cot if available (only in CoT data)
    if 'inject_cot' in cot_df.columns:
        result_cols.append('inject_cot')
    
    # Handle missing columns gracefully
    if 'question_zs' not in merged.columns:
        if 'question' in merged.columns:
            merged['question_zs'] = merged['question']
        else:
            merged['question_zs'] = 'N/A'
    
    # Handle response columns
    for col in ['response_zs', 'response_cot']:
        if col not in merged.columns:
            base_col = col.replace('_zs', '').replace('_cot', '')
            if base_col in merged.columns:
                merged[col] = merged[base_col]
            else:
                merged[col] = 'N/A'
    
    # Select available columns
    available_cols = [col for col in result_cols if col in merged.columns]
    result = merged[available_cols].copy()
    
    return result


def curate_top_shifts(model_name: str, datasets: List[str] = None, 
                     top_k: int = 200, output_file: str = None) -> Dict[str, pd.DataFrame]:
    """Curate top confidence shifts for a given model across datasets
    
    Args:
        model_name: Name of the model to analyze
        datasets: List of datasets to analyze (None for all available)
        top_k: Number of top examples to return per dataset
        output_file: Path to save combined results JSON
    
    Returns:
        Dictionary mapping dataset names to DataFrames of top shifts
    """
    print(f"🔍 Discovering files for model: {model_name}")
    
    # Discover all files for this model
    model_files = discover_model_files(model_name)
    
    if not model_files:
        print(f"❌ No files found for model: {model_name}")
        return {}
    
    print(f"📊 Found data for datasets: {list(model_files.keys())}")
    
    # Filter datasets if specified
    if datasets:
        model_files = {k: v for k, v in model_files.items() if k in datasets}
        if not model_files:
            print(f"❌ No files found for specified datasets: {datasets}")
            return {}
    
    results = {}
    all_results = []
    
    for dataset, files in model_files.items():
        print(f"\n📈 Processing dataset: {dataset.upper()}")
        
        # Check if we have both ZS and CoT files
        if 'zs' not in files or 'cot' not in files:
            print(f"⚠️  Missing ZS or CoT data for {dataset}, skipping...")
            continue
        
        # Load the data
        print(f"   Loading ZS data from: {files['zs']}")
        zs_df = load_json_data(files['zs'])
        
        print(f"   Loading CoT data from: {files['cot']}")
        cot_df = load_json_data(files['cot'])
        
        if zs_df is None or cot_df is None:
            print(f"❌ Failed to load data for {dataset}")
            continue
        
        print(f"   ZS examples: {len(zs_df)}, CoT examples: {len(cot_df)}")
        
        # Calculate confidence shifts
        shifts_df = calculate_confidence_shifts(zs_df, cot_df)
        
        if len(shifts_df) == 0:
            print(f"❌ No matching examples found for {dataset}")
            continue
        
        print(f"   Matching examples: {len(shifts_df)}")
        
        # Sort by absolute confidence shift and take top K
        top_shifts = shifts_df.nlargest(top_k, 'confidence_shift_abs').copy()
        top_shifts['dataset'] = dataset
        top_shifts['model'] = model_name
        
        results[dataset] = top_shifts
        all_results.append(top_shifts)
        
        # Print summary statistics
        print(f"   📊 Top {min(top_k, len(top_shifts))} shifts summary:")
        print(f"      Mean absolute shift: {top_shifts['confidence_shift_abs'].mean():.4f}")
        print(f"      Max absolute shift: {top_shifts['confidence_shift_abs'].max():.4f}")
        print(f"      Accuracy improvements: {(top_shifts['accuracy_change'] > 0).sum()}")
        print(f"      Accuracy degradations: {(top_shifts['accuracy_change'] < 0).sum()}")
        
        # Show shift categories
        category_counts = top_shifts['shift_category'].value_counts()
        for category, count in category_counts.items():
            print(f"      {category}: {count}")
    
    # Save combined results if output file specified
    if output_file and all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Convert categorical columns to strings for JSON serialization
        categorical_cols = ['zs_confidence_level', 'cot_confidence_level']
        for col in categorical_cols:
            if col in combined_df.columns:
                combined_df[col] = combined_df[col].astype(str)
        
        # Create structured JSON output
        examples_by_dataset = {}
        for dataset in combined_df['dataset'].unique():
            dataset_examples = combined_df[combined_df['dataset'] == dataset].copy()
            dataset_examples = dataset_examples.drop(['dataset', 'model'], axis=1)
            examples_by_dataset[dataset] = {
                "count": len(dataset_examples),
                "mean_abs_shift": float(dataset_examples['confidence_shift_abs'].mean()),
                "max_abs_shift": float(dataset_examples['confidence_shift_abs'].max()),
                "accuracy_improvements": int((dataset_examples['accuracy_change'] > 0).sum()),
                "accuracy_degradations": int((dataset_examples['accuracy_change'] < 0).sum()),
                "examples": dataset_examples.to_dict('records')
            }
        
        # Convert DataFrame to structured JSON
        json_data = {
            "metadata": {
                "model": model_name,
                "total_examples": len(combined_df),
                "datasets_analyzed": list(combined_df['dataset'].unique()),
                "top_k_per_dataset": top_k,
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
                "overall_statistics": {
                    "mean_abs_shift": float(combined_df['confidence_shift_abs'].mean()),
                    "median_abs_shift": float(combined_df['confidence_shift_abs'].median()),
                    "max_abs_shift": float(combined_df['confidence_shift_abs'].max()),
                    "min_abs_shift": float(combined_df['confidence_shift_abs'].min()),
                    "confidence_increases": int((combined_df['confidence_shift'] > 0).sum()),
                    "confidence_decreases": int((combined_df['confidence_shift'] < 0).sum()),
                    "accuracy_improvements": int((combined_df['accuracy_change'] > 0).sum()),
                    "accuracy_degradations": int((combined_df['accuracy_change'] < 0).sum()),
                    "shift_categories": combined_df['shift_category'].value_counts().to_dict()
                }
            },
            "datasets": examples_by_dataset
        }
        
        # Save as JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Saved combined results to: {output_file}")
        print(f"   Total examples: {len(combined_df)}")
        print(f"   Datasets: {combined_df['dataset'].nunique()}")
    
    return results


def print_summary_statistics(results: Dict[str, pd.DataFrame], model_name: str):
    """Print comprehensive summary statistics"""
    if not results:
        return
    
    print(f"\n📊 SUMMARY STATISTICS FOR {model_name.upper()}")
    print("=" * 60)
    
    total_examples = sum(len(df) for df in results.values())
    print(f"Total curated examples: {total_examples}")
    print(f"Datasets analyzed: {len(results)}")
    
    # Combine all results for overall statistics
    all_data = pd.concat(results.values(), ignore_index=True)
    
    print(f"\nOverall Confidence Shift Statistics:")
    print(f"  Mean absolute shift: {all_data['confidence_shift_abs'].mean():.4f}")
    print(f"  Median absolute shift: {all_data['confidence_shift_abs'].median():.4f}")
    print(f"  Max absolute shift: {all_data['confidence_shift_abs'].max():.4f}")
    print(f"  Min absolute shift: {all_data['confidence_shift_abs'].min():.4f}")
    
    print(f"\nConfidence Shift Direction:")
    positive_shifts = (all_data['confidence_shift'] > 0).sum()
    negative_shifts = (all_data['confidence_shift'] < 0).sum()
    print(f"  CoT increased confidence: {positive_shifts} ({positive_shifts/len(all_data)*100:.1f}%)")
    print(f"  CoT decreased confidence: {negative_shifts} ({negative_shifts/len(all_data)*100:.1f}%)")
    
    print(f"\nAccuracy Changes:")
    improvements = (all_data['accuracy_change'] > 0).sum()
    degradations = (all_data['accuracy_change'] < 0).sum()
    no_change = (all_data['accuracy_change'] == 0).sum()
    print(f"  Improved (incorrect → correct): {improvements} ({improvements/len(all_data)*100:.1f}%)")
    print(f"  Degraded (correct → incorrect): {degradations} ({degradations/len(all_data)*100:.1f}%)")
    print(f"  No change: {no_change} ({no_change/len(all_data)*100:.1f}%)")
    
    print(f"\nShift Categories:")
    category_counts = all_data['shift_category'].value_counts()
    for category, count in category_counts.items():
        percentage = count / len(all_data) * 100
        print(f"  {category.replace('_', ' ').title()}: {count} ({percentage:.1f}%)")
    
    print(f"\nPer-Dataset Breakdown:")
    for dataset, df in results.items():
        print(f"  {dataset.upper()}:")
        print(f"    Examples: {len(df)}")
        print(f"    Mean abs shift: {df['confidence_shift_abs'].mean():.4f}")
        print(f"    Accuracy improvements: {(df['accuracy_change'] > 0).sum()}")


def main():
    parser = argparse.ArgumentParser(
        description="Curate examples with largest confidence shifts between Zero-Shot and CoT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all datasets for a specific model
  python curate_confidence_shifts.py --model "gemma-2-9b-it"
  
  # Analyze specific dataset with custom output
  python curate_confidence_shifts.py --model " Meta-Llama-31-8B-Instruct" --dataset gsm8k --output llama_gsm8k_shifts.json
  
  # Get top 100 examples instead of default 200
  python curate_confidence_shifts.py --model "Qwen25-7B-Instruct" --top-k 100
        """
    )
    
    parser.add_argument('--model', '-m', required=True,
                       help='Model name to analyze (must match filename pattern)')
    parser.add_argument('--dataset', '-d', action='append',
                       help='Dataset(s) to analyze (can be specified multiple times). If not provided, analyzes all available datasets.')
    parser.add_argument('--top-k', '-k', type=int, default=200,
                       help='Number of top examples to return per dataset (default: 200)')
    parser.add_argument('--output', '-o', 
                       help='Output JSON file path. If not provided, uses model_name_confidence_shifts.json')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only print summary statistics without saving detailed results')
    
    args = parser.parse_args()
    
    # Set default output filename if not provided
    if not args.output and not args.summary_only:
        safe_model_name = args.model.replace('/', '_').replace('\\', '_')
        args.output = f"{safe_model_name}_confidence_shifts.json"
    
    # Run the analysis
    results = curate_top_shifts(
        model_name=args.model,
        datasets=args.dataset,
        top_k=args.top_k,
        output_file=args.output if not args.summary_only else None
    )
    
    # Print summary statistics
    print_summary_statistics(results, args.model)
    
    if not results:
        print(f"\n❌ No results found for model: {args.model}")
        if args.dataset:
            print(f"   Requested datasets: {args.dataset}")
        print("\n💡 Tips:")
        print("   - Check that the model name exactly matches the filename pattern")
        print("   - Ensure both cot_exp and zs_exp directories exist")
        print("   - Verify that JSON files exist for both experiment types")
        return
    
    print(f"\n✅ Analysis complete for {args.model}")
    if not args.summary_only:
        print(f"📁 Detailed results saved to: {args.output}")


if __name__ == "__main__":
    main() 