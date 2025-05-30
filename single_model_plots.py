"""Single model analysis plots for model confidence calibration"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from typing import Dict

from plotting_utils import (
    SUMMARY_PLOT_SIZE, LARGE_PLOT_SIZE, MODEL_NAME_MAPPING,
    EXP_TYPE_MAPPING, apply_clean_style, create_summary_dataframe
)


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
        print(f"     - {model_name}_{dataset.lower()}.pdf/png")
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