"""Summary plots for model confidence calibration analysis"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from typing import Dict

from plotting_utils import (
    SUMMARY_PLOT_SIZE, LARGE_PLOT_SIZE, MODEL_NAME_MAPPING,
    EXP_TYPE_MAPPING, apply_clean_style, create_summary_dataframe
)


def create_summary_plots(all_data: Dict, save_dir: str = "summary_plots"):
    """Create summary plots across models and datasets, separated by experiment type"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create summary dataframe
    summary_df = create_summary_dataframe(all_data)
    
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