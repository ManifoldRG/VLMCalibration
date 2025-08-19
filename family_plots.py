"""Family-wise analysis plots for model confidence calibration"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
import glob
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

from plotting_utils import (
    SUMMARY_PLOT_SIZE, LARGE_PLOT_SIZE, MODEL_NAME_MAPPING,
    EXP_TYPE_MAPPING, FAMILY_COLORS, apply_clean_style, 
    categorize_model_family, create_summary_dataframe,
    discover_data_files, parse_filename, load_json_data
)


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
    
    print(f"🏠 Creating family-wise confidence distribution analysis...")
    print(f"   Datasets: {', '.join(datasets)}")
    print(f"   Experiment types: {', '.join([EXP_TYPE_MAPPING.get(exp, exp) for exp in exp_types])}")
    print(f"   Quantized models: {'Included' if quant else 'Excluded'}")
    
    # Create confidence distribution plots for each dataset and experiment type
    created_plots = []
    
    for dataset in datasets:
        for exp_type in exp_types:
            exp_name = EXP_TYPE_MAPPING.get(exp_type, exp_type)
            print(f"   📊 Processing {dataset.upper()} - {exp_name}...")
            
            # Create confidence distribution plots for this dataset/experiment
            result = create_family_confidence_distributions(
                all_data, dataset, exp_type, FAMILY_COLORS, save_dir
            )
            
            if result:
                filename_correct, filename_incorrect, filename_combined = result
                created_plots.extend([
                    (dataset, exp_type, 'correct', filename_correct),
                    (dataset, exp_type, 'incorrect', filename_incorrect),
                    (dataset, exp_type, 'combined', filename_combined)
                ])
    
    # Create overall summary statistics across all tasks
    create_family_summary_statistics(all_data, save_dir, FAMILY_COLORS, quant)
    
    # Create family calibration plots
    summary_df = create_summary_dataframe(all_data)
    create_family_calibration_plots(summary_df, save_dir)
    
    print(f"📊 Family confidence distribution analysis complete!")
    print(f"   Results saved in '{save_dir}/' directory:")
    
    # Group and display created plots
    for dataset in datasets:
        for exp_type in exp_types:
            # Create shorter filename suffix based on experiment type
            if exp_type == "cot_exp":
                exp_short = "cot"
            elif exp_type == "zs_exp":
                exp_short = "zs"
            elif exp_type == "verbalized":
                exp_short = "verbalized"
            elif exp_type == "verbalized_cot":
                exp_short = "verbalized_cot"
            elif exp_type == "otherAI":
                exp_short = "otherAI"
            elif exp_type == "otherAI_cot":
                exp_short = "otherAI_cot"
            elif exp_type == "otherAI_verbalized":
                exp_short = "otherAI_verbalized"
            elif exp_type == "otherAI_verbalized_cot":
                exp_short = "otherAI_verbalized_cot"
            else:
                exp_short = exp_type
            
            exp_name = EXP_TYPE_MAPPING.get(exp_type, exp_type)
            print(f"   • {dataset.upper()} ({exp_name}):")
            print(f"     - family_confidence_correct_{dataset.lower()}_{exp_short}.pdf/png")
            print(f"     - family_confidence_incorrect_{dataset.lower()}_{exp_short}.pdf/png")
            print(f"     - family_confidence_combined_{dataset.lower()}_{exp_short}.pdf/png")
            print(f"     - family_confidence_stats_{dataset.lower()}_{exp_short}.csv")
    
    print(f"   • Overall Analysis:")
    print(f"     - family_summary_statistics.csv")
    print(f"     - family_confidence_overview.pdf/png")


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
                
                color = FAMILY_COLORS.get(family, '#888888')
                
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
    create_overall_family_comparison(summary_df, family_dir, FAMILY_COLORS)
    
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


def create_family_summary_statistics(all_data: Dict, save_dir: str, 
                                   family_colors: Dict[str, str], quant: bool):
    """Create overall summary statistics and overview plot for family analysis"""
    
    # Create summary dataframe
    summary_df = create_summary_dataframe(all_data)
    
    if len(summary_df) == 0:
        print("❌ No data available for family summary statistics")
        return
    
    # Add family column
    summary_df['family'] = summary_df['model'].apply(categorize_model_family)
    
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
    exp_name = EXP_TYPE_MAPPING.get(exp_type, exp_type)
    
    # Create shorter filename suffix based on experiment type
    if exp_type == "cot_exp":
        exp_short = "cot"
    elif exp_type == "zs_exp":
        exp_short = "zs"
    elif exp_type == "verbalized":
        exp_short = "verbalized"
    elif exp_type == "verbalized_cot":
        exp_short = "verbalized_cot"
    elif exp_type == "otherAI":
        exp_short = "otherAI"
    elif exp_type == "otherAI_cot":
        exp_short = "otherAI_cot"
    elif exp_type == "otherAI_verbalized":
        exp_short = "otherAI_verbalized"
    elif exp_type == "otherAI_verbalized_cot":
        exp_short = "otherAI_verbalized_cot"
    else:
        exp_short = exp_type
    
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


def load_experiment_results(base_dir: str = ".", quant: bool = False) -> Dict:
    """Automatically discover and load all experiment results from the directory structure
    
    Args:
        base_dir: Base directory to search for experiment results
        quant: If True, only include quantized models. If False, only include non-quantized models.
        
    Returns:
        Dictionary with experiment data in the format expected by plotting functions
    """
    all_data = {}
    
    if quant:
        # For quantized models, search specifically in archive directories
        base_dirs = []
        archive_experiments = ['cot_exp', 'zs_exp', 'verbalized', 'verbalized_cot', 'otherAI', 'otherAI_cot', 'otherAI_verbalized', 'otherAI_verbalized_cot']
        
        for exp_type in archive_experiments:
            archive_path = os.path.join(base_dir, 'archive', exp_type)
            if os.path.exists(archive_path):
                base_dirs.append(archive_path)
        
        # Also check if there are archive patterns directly
        archive_patterns = [
            os.path.join(base_dir, 'archive'),
            os.path.join(base_dir, 'train_results_archive')
        ]
        for pattern in archive_patterns:
            if os.path.exists(pattern):
                # Find subdirectories that match experiment types
                for item in os.listdir(pattern):
                    item_path = os.path.join(pattern, item)
                    if os.path.isdir(item_path) and item in archive_experiments:
                        base_dirs.append(item_path)
        
        if not base_dirs:
            print(f"❌ No archive directories found for quantized models in {base_dir}")
            return {}
        
        print(f"🔍 Searching for quantized models in archive directories: {base_dirs}")
        
        # Manually discover GGUF files in archive directories
        all_files = []
        for archive_dir in base_dirs:
            pattern = os.path.join(archive_dir, "*", "*_records_*.json")
            files = glob.glob(pattern)
            
            # Filter to only GGUF files and exclude experiment_details
            gguf_files = [f for f in files if 'gguf' in os.path.basename(f).lower() and 'experiment_details' not in os.path.basename(f)]
            all_files.extend(gguf_files)
            
        print(f"🔍 Found {len(all_files)} GGUF files in archive directories")
        
    else:
        # Define the experiment types we're looking for (same as generate_all_plots.py and summary_plots.py)
        base_dirs = ['cot_exp', 'zs_exp', 'verbalized', 'verbalized_cot', 'otherAI', 'otherAI_cot']
        
        # Discover data files - exclude archive folders for non-quantized models
        data_files = discover_data_files(base_dirs, quant=False)
        
        if not data_files:
            print("❌ No data files found!")
            return {}
        
        # Collect all file paths and filter out archive and tracing folders
        all_files = []
        for base_dir_name, files in data_files.items():
            # Filter out both archive and tracing folders for non-quantized models
            filtered_files = [f for f in files if "tracing" not in f.lower() and "archive" not in f.lower()]
            all_files.extend(filtered_files)
        
        # Only keep non-quantized models (exclude GGUF files)
        all_files = [f for f in all_files if not ('gguf' in f.lower() or 'q4' in f.lower() or 'q8' in f.lower())]
        print(f"🔍 Found {len(all_files)} non-quantized model files (excluding archive and tracing folders)")
    
    if not all_files:
        model_type = "quantized" if quant else "non-quantized"
        print(f"❌ No {model_type} model files found!")
        return {}
    
    # Process files
    for filepath in all_files:
        try:
            metadata = parse_filename(filepath)
            
            # Skip train files
            if metadata['split'] == 'train':
                continue
                
            df = load_json_data(filepath, sample=False)
            if df is None:
                continue
            
            # Create key for all_data dictionary
            key = f"{metadata['exp_type']}_{metadata['dataset']}_{metadata['split']}_{metadata['model_name']}"
            all_data[key] = (df, metadata)
            
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            continue
    
    model_type = "quantized" if quant else "non-quantized"
    print(f"✅ Successfully loaded {len(all_data)} {model_type} model experiments")
    return all_data


def print_experiment_summary(all_data: Dict):
    """Print a summary of loaded experiments"""
    
    if not all_data:
        print("❌ No experiment data loaded")
        return
        
    print(f"\n📊 Family Analysis - Experiment Summary:")
    print(f"   Total experiments: {len(all_data)}")
    
    # Group by experiment type
    exp_types = {}
    datasets = set()
    models = set()
    
    for key, (df, metadata) in all_data.items():
        exp_type = metadata['exp_type']
        dataset = metadata['dataset']
        model = metadata['model_name']
        
        if exp_type not in exp_types:
            exp_types[exp_type] = []
        exp_types[exp_type].append(metadata)
        
        datasets.add(dataset)
        models.add(model)
    
    print(f"   Experiment types: {list(exp_types.keys())}")
    print(f"   Datasets: {sorted(datasets)}")
    print(f"   Models: {len(models)} unique models")
    
    print(f"\n📋 Detailed breakdown:")
    for exp_type, experiments in exp_types.items():
        exp_name = EXP_TYPE_MAPPING.get(exp_type, exp_type)
        print(f"   • {exp_name} ({exp_type}): {len(experiments)} experiments")
        
        # Group by dataset
        dataset_counts = {}
        for exp in experiments:
            dataset = exp['dataset']
            if dataset not in dataset_counts:
                dataset_counts[dataset] = 0
            dataset_counts[dataset] += 1
        
        for dataset, count in sorted(dataset_counts.items()):
            print(f"     - {dataset}: {count} model(s)")


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="Generate family-wise analysis plots for model confidence calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all non-quantized experiments in current directory
  python family_plots.py
  
  # Analyze experiments in specific directory
  python family_plots.py --base_dir /path/to/experiments
  
  # Analyze only quantized models
  python family_plots.py --quant
  
  # Save plots to specific directory
  python family_plots.py --output_dir my_family_analysis
  
  # Only show summary without generating plots
  python family_plots.py --summary_only
        """
    )
    
    parser.add_argument(
        "--base_dir",
        type=str,
        default=".",
        help="Base directory to search for experiment results (default: current directory)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str,
        default="family_analysis",
        help="Directory to save output plots and analysis (default: family_analysis)"
    )
    
    parser.add_argument(
        "--quant",
        action="store_true",
        help="Analyze only GGUF quantized models (default: analyze only non-quantized models)"
    )
    
    parser.add_argument(
        "--summary_only",
        action="store_true",
        help="Only print experiment summary without generating plots"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true", 
        help="Print verbose output during processing"
    )
    
    args = parser.parse_args()
    
    print("🏠 Model Family Confidence Analysis")
    print("=" * 50)
    
    if args.quant:
        print("⚖️  Analyzing only GGUF quantized models")
    else:
        print("🚫 Analyzing only non-quantized models (use --quant for quantized models only)")
    
    # Load experiment results
    print(f"📂 Searching for experiments in: {os.path.abspath(args.base_dir)}")
    all_data = load_experiment_results(args.base_dir, quant=args.quant)
    
    if not all_data:
        model_type = "quantized" if args.quant else "non-quantized"
        print(f"❌ No {model_type} experiment results found!")
        print("\nMake sure you have run experiments and have the expected directory structure:")
        print("  {exp_type}/{dataset}/{exp_type}_records_{split}_{model_name}.json")
        print("\nSupported experiment types:")
        print("  • cot_exp (Chain-of-Thought)")
        print("  • zs_exp (Zero-Shot)")
        print("  • verbalized (Verbalized Confidence)")
        print("  • verbalized_cot (Verbalized Confidence with CoT)")
        print("  • otherAI (Other AI Evaluation)")
        print("  • otherAI_cot (Other AI Evaluation with CoT)")
        if args.quant:
            print("\nNote: Looking for quantized model files (containing 'gguf', 'q4', or 'q8')")
        else:
            print("\nNote: Excluding quantized model files. Use --quant to analyze quantized models only.")
        return
    
    # Print summary
    print_experiment_summary(all_data)
    
    if args.summary_only:
        model_type = "quantized" if args.quant else "non-quantized"
        print(f"\n✅ Summary complete for {model_type} models. Use without --summary_only to generate family plots.")
        return
    
    # Generate family plots
    print(f"\n🎨 Generating family analysis plots...")
    print(f"   Output directory: {os.path.abspath(args.output_dir)}")
    
    try:
        plots_across_families(all_data, args.output_dir, quant=args.quant)
        model_type = "quantized" if args.quant else "non-quantized"
        print(f"\n🎉 Family analysis complete for {model_type} models! Check '{args.output_dir}' for results.")
        
        # Print quick access info
        print(f"\n📈 Key outputs:")
        print(f"   • Family summary statistics: {args.output_dir}/family_summary_statistics.csv")
        print(f"   • Family confidence overview: {args.output_dir}/family_confidence_overview.pdf")
        print(f"   • Individual family plots by dataset and experiment type")
        print(f"   • Family calibration analysis: {args.output_dir}/family_plots/")
        
    except Exception as e:
        print(f"❌ Error generating family plots: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return
    
    print(f"\n✅ Family analysis complete! 🏠")


if __name__ == "__main__":
    main() 