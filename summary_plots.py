"""Summary plots for model confidence calibration analysis"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from typing import Dict
import seaborn as sns
import alphashape

from plotting_utils import (
    SUMMARY_PLOT_SIZE, LARGE_PLOT_SIZE, MODEL_NAME_MAPPING,
    EXP_TYPE_MAPPING, apply_clean_style, create_summary_dataframe
)


def calculate_ece(df: pd.DataFrame, n_bins: int = 10) -> float:
    """Calculate Expected Calibration Error (ECE)
    
    Args:
        df: DataFrame with 'p_true' (confidence) and 'correct' (boolean) columns
        n_bins: Number of bins for calibration calculation
        
    Returns:
        ECE value (lower is better, 0 is perfect calibration)
    """
    if len(df) == 0:
        return np.nan
        
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0
    
    for i in range(n_bins):
        # Define bin boundaries
        if i == n_bins - 1:  # Last bin includes 1.0
            mask = (df['p_true'] >= bin_edges[i]) & (df['p_true'] <= bin_edges[i+1])
        else:
            mask = (df['p_true'] >= bin_edges[i]) & (df['p_true'] < bin_edges[i+1])
        
        if mask.sum() > 0:
            bin_acc = df[mask]['correct'].mean()
            bin_conf = df[mask]['p_true'].mean()
            bin_weight = mask.sum() / len(df)
            ece += bin_weight * abs(bin_acc - bin_conf)
    
    return ece


def calculate_brier_score(df: pd.DataFrame) -> float:
    """Calculate Brier Score (combines accuracy and calibration)
    
    Args:
        df: DataFrame with 'p_true' (confidence) and 'correct' (boolean) columns
        
    Returns:
        Brier score (lower is better, 0 is perfect)
    """
    if len(df) == 0:
        return np.nan
    
    return np.mean((df['p_true'] - df['correct'].astype(float)) ** 2)


def calculate_overconfidence_rate(df: pd.DataFrame, threshold: float = 0.05) -> float:
    """Calculate rate of overconfident predictions
    
    Args:
        df: DataFrame with 'p_true' (confidence) and 'correct' (boolean) columns
        threshold: Minimum difference to consider overconfident
        
    Returns:
        Fraction of predictions where confidence > accuracy + threshold
    """
    if len(df) == 0:
        return np.nan
    
    # For each prediction, compare confidence to accuracy
    # This is a simplified version - proper overconfidence needs binning
    accuracy = df['correct'].mean()
    overconfident_mask = df['p_true'] > (accuracy + threshold)
    return overconfident_mask.mean()


def create_enhanced_summary_dataframe(all_data: Dict) -> pd.DataFrame:
    """Create enhanced summary dataframe with proper calibration metrics"""
    
    summary_data = []
    
    for key, df_info in all_data.items():
        df, metadata = df_info
        if df is None:
            continue
            
        correct_data = df[df['correct'] == True]['p_true']
        incorrect_data = df[df['correct'] == False]['p_true']
        
        # Calculate enhanced metrics
        ece = calculate_ece(df)
        brier_score = calculate_brier_score(df)
        overconf_rate = calculate_overconfidence_rate(df)
        
        summary_data.append({
            'model': metadata['model_name'],
            'dataset': metadata['dataset'], 
            'split': metadata['split'],
            'exp_type': metadata['exp_type'],
            'accuracy': (df['correct'] == True).mean(),
            'ece': ece,
            'brier_score': brier_score,
            'overconfidence_rate': overconf_rate,
            'mean_conf_correct': correct_data.mean(),
            'mean_conf_incorrect': incorrect_data.mean(),
            'confidence_gap': correct_data.mean() - incorrect_data.mean(),  # Keep for backward compatibility
            'n_samples': len(df)
        })
    
    return pd.DataFrame(summary_data)


def create_performance_landscape_with_alphashapes(summary_df: pd.DataFrame, exp_type: str, save_dir: str):
    """Create accuracy vs ECE landscape with alpha shapes showing performance regions"""
    
    exp_data = summary_df[summary_df['exp_type'] == exp_type].copy()
    
    if len(exp_data) == 0:
        return
        
    exp_name = EXP_TYPE_MAPPING[exp_type]
    exp_short = "cot" if exp_type == "cot_exp" else "zs"
    
    # Create the plot
    fig, ax = plt.subplots(figsize=LARGE_PLOT_SIZE)
    
    # Get unique datasets and models for coloring
    datasets = sorted(exp_data['dataset'].unique())
    models = sorted(exp_data['model'].unique())
    
    # Create color palette for datasets
    dataset_colors = plt.cm.Set1(np.linspace(0, 1, len(datasets)))
    dataset_color_map = {dataset: dataset_colors[i] for i, dataset in enumerate(datasets)}
    
    # Create marker styles for models
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '8']
    model_marker_map = {model: markers[i % len(markers)] for i, model in enumerate(models)}
    
    # Plot density background using hexbin
    x_data = exp_data['accuracy'].values
    y_data = exp_data['ece'].values
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
    x_data = x_data[valid_mask]
    y_data = y_data[valid_mask]
    exp_data_clean = exp_data[valid_mask]
    
    if len(x_data) > 5:  # Need minimum points for hexbin
        hb = ax.hexbin(x_data, y_data, gridsize=15, cmap='Blues', alpha=0.3, 
                       extent=[0, 1, 0, max(0.5, y_data.max() * 1.1)])
        
        # Add alpha shapes if available and we have enough points
        if len(x_data) >= 4:
            try:
                points = np.column_stack([x_data, y_data])
                
                # Create alpha shape for overall performance boundary
                alpha_shape = alphashape.alphashape(points, alpha=0.1)
                
                if hasattr(alpha_shape, 'exterior'):
                    # It's a polygon
                    x_boundary, y_boundary = alpha_shape.exterior.xy
                    ax.plot(x_boundary, y_boundary, 'k--', alpha=0.6, linewidth=2, 
                           label='Performance Boundary')
                elif hasattr(alpha_shape, 'geoms'):
                    # It's a collection of geometries
                    for geom in alpha_shape.geoms:
                        if hasattr(geom, 'exterior'):
                            x_boundary, y_boundary = geom.exterior.xy
                            ax.plot(x_boundary, y_boundary, 'k--', alpha=0.6, linewidth=2)
                
            except Exception as e:
                print(f"⚠️  Alpha shape calculation failed: {e}")
    
    # Plot individual points
    for _, row in exp_data_clean.iterrows():
        color = dataset_color_map[row['dataset']]
        marker = model_marker_map[row['model']]
        
        ax.scatter(row['accuracy'], row['ece'], 
                  c=[color], marker=marker, s=80, alpha=0.8, 
                  edgecolors='black', linewidth=0.5)
    
    # Add "ideal performance" indicators
    # Perfect calibration line (ECE = 0)
    ax.axhline(y=0, color='green', linestyle='-', alpha=0.5, linewidth=2, 
               label='Perfect Calibration (ECE=0)')
    
    # Good performance region (high accuracy, low ECE)
    if len(y_data) > 0:
        good_ece_threshold = np.percentile(y_data, 25)  # Bottom quartile of ECE
        good_acc_threshold = np.percentile(x_data, 75)  # Top quartile of accuracy
        
        # Highlight good performance region
        ax.axvspan(good_acc_threshold, 1, ymin=0, ymax=good_ece_threshold/ax.get_ylim()[1], 
                   alpha=0.2, color='green', label=f'High Performance Region')
    
    # Create legends
    # Dataset legend (colors)
    dataset_handles = []
    for dataset in datasets:
        handle = plt.Line2D([0], [0], marker='o', color='w', 
                           markerfacecolor=dataset_color_map[dataset], 
                           markersize=8, alpha=0.8, markeredgecolor='black',
                           markeredgewidth=0.5, label=dataset.upper())
        dataset_handles.append(handle)
    
    # Model legend (markers) - show only first few to avoid clutter
    model_handles = []
    for i, model in enumerate(models[:8]):  # Limit to first 8 models
        # Use MODEL_NAME_MAPPING for consistent display names
        model_display = MODEL_NAME_MAPPING.get(model, model)
        if len(model_display) > 15:
            model_display = model_display[:12] + '...'
        
        handle = plt.Line2D([0], [0], marker=model_marker_map[model], color='w',
                           markerfacecolor='gray', markersize=8, alpha=0.8,
                           markeredgecolor='black', markeredgewidth=0.5,
                           label=model_display)
        model_handles.append(handle)
    
    if len(models) > 8:
        model_handles.append(plt.Line2D([0], [0], marker='', color='w', 
                                       label=f'... and {len(models)-8} more'))
    
    # Add legends
    dataset_legend = ax.legend(handles=dataset_handles, loc='upper left', 
                              title='Datasets', frameon=True, fancybox=False,
                              edgecolor='#cccccc', facecolor='white', 
                              framealpha=0.9, title_fontsize=10)
    dataset_legend.get_frame().set_linewidth(0.5)
    
    # Add performance indicators legend
    perf_handles = [plt.Line2D([0], [0], color='green', linestyle='-', alpha=0.5, 
                               linewidth=2, label='Perfect Calibration')]
    perf_handles.append(plt.Line2D([0], [0], color='k', linestyle='--', alpha=0.6, 
                                    linewidth=2, label='Performance Boundary'))
    
    perf_legend = ax.legend(handles=perf_handles, loc='upper right', 
                           title='Performance Indicators', frameon=True, fancybox=False,
                           edgecolor='#cccccc', facecolor='white', framealpha=0.9)
    perf_legend.get_frame().set_linewidth(0.5)
    ax.add_artist(dataset_legend)  # Keep both legends
    
    apply_clean_style(ax)
    ax.set_xlabel('Accuracy', fontweight='normal')
    ax.set_ylabel('Expected Calibration Error (ECE)', fontweight='normal')
    ax.set_title(f'Model Performance Landscape - {exp_name}\nAccuracy vs Calibration Quality', 
                 fontweight='normal', pad=15)
    
    # Set reasonable axis limits
    ax.set_xlim(0, 1)
    if len(y_data) > 0:
        ax.set_ylim(0, max(0.3, y_data.max() * 1.1))
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{exp_short}_performance_landscape.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, f'{exp_short}_performance_landscape.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def create_cross_dataset_consistency(summary_df: pd.DataFrame, save_dir: str):
    """Create performance profiles showing model consistency across datasets"""
    
    datasets = sorted(summary_df['dataset'].unique())
    exp_types = sorted(summary_df['exp_type'].unique())
    
    if len(datasets) < 2:
        return
    
    fig, axes = plt.subplots(len(exp_types), len(datasets), 
                           figsize=(4 * len(datasets), 4 * len(exp_types)))
    
    if len(exp_types) == 1:
        axes = axes.reshape(1, -1)
    if len(datasets) == 1:
        axes = axes.reshape(-1, 1)
    
    for exp_idx, exp_type in enumerate(exp_types):
        exp_name = EXP_TYPE_MAPPING[exp_type]
        exp_data = summary_df[summary_df['exp_type'] == exp_type]
        
        for dataset_idx, dataset in enumerate(datasets):
            ax = axes[exp_idx, dataset_idx]
            dataset_data = exp_data[exp_data['dataset'] == dataset]
            
            if len(dataset_data) > 0:
                # Create density plot
                x_data = dataset_data['accuracy'].values
                y_data = dataset_data['ece'].values
                
                # Remove NaN values
                valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
                x_data = x_data[valid_mask]
                y_data = y_data[valid_mask]
                
                if len(x_data) > 1:
                    # Use seaborn for better density visualization
                    sns.scatterplot(data=dataset_data[valid_mask], x='accuracy', y='ece', 
                                  alpha=0.7, ax=ax)
                    
                    # Add trend line if enough points
                    if len(x_data) > 5:
                        sns.regplot(data=dataset_data[valid_mask], x='accuracy', y='ece', 
                                  scatter=False, ax=ax, color='red', alpha=0.5)
                
                # Add perfect calibration reference
                ax.axhline(y=0, color='green', linestyle='--', alpha=0.5, linewidth=1)
            
            apply_clean_style(ax)
            ax.set_title(f'{dataset.upper()}\n{exp_name}', fontweight='normal', fontsize=10)
            ax.set_xlabel('Accuracy' if exp_idx == len(exp_types)-1 else '', fontweight='normal')
            ax.set_ylabel('ECE' if dataset_idx == 0 else '', fontweight='normal')
            ax.set_xlim(0, 1)
            
            # Set consistent y-limits across subplots
            if len(summary_df) > 0:
                max_ece = summary_df['ece'].quantile(0.95)  # Use 95th percentile to avoid outliers
                ax.set_ylim(0, max(0.2, max_ece))
    
    fig.suptitle('Model Performance Consistency Across Datasets', 
                 fontweight='bold', fontsize=14, y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    fig.savefig(os.path.join(save_dir, 'cross_dataset_consistency.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, 'cross_dataset_consistency.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def create_calibration_metrics_comparison(summary_df: pd.DataFrame, save_dir: str):
    """Create enhanced CoT vs ZS comparison using proper calibration metrics"""
    
    # Separate CoT and ZS data
    cot_data = summary_df[summary_df['exp_type'] == 'cot_exp'].copy()
    zs_data = summary_df[summary_df['exp_type'] == 'zs_exp'].copy()
    
    if len(cot_data) == 0 or len(zs_data) == 0:
        print("⚠️  Cannot create calibration metrics comparison - missing data for one experiment type")
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
            
            # Calculate improvements (CoT - ZS, negative ECE improvement is better)
            accuracy_improvement = cot_row['accuracy'] - zs_row['accuracy']
            ece_improvement = zs_row['ece'] - cot_row['ece']  # Negative ECE change is improvement
            brier_improvement = zs_row['brier_score'] - cot_row['brier_score']  # Negative change is improvement
            
            comparison_data.append({
                'model': cot_row['model'],
                'dataset': cot_row['dataset'],
                'split': cot_row['split'],
                'model_dataset': f"{cot_row['model']}_{cot_row['dataset']}",
                'accuracy_improvement': accuracy_improvement,
                'ece_improvement': ece_improvement,
                'brier_improvement': brier_improvement,
                'cot_accuracy': cot_row['accuracy'],
                'zs_accuracy': zs_row['accuracy'],
                'cot_ece': cot_row['ece'],
                'zs_ece': zs_row['ece'],
                'cot_brier': cot_row['brier_score'],
                'zs_brier': zs_row['brier_score']
            })
    
    if len(comparison_data) == 0:
        print("⚠️  No matching CoT-ZS pairs found for comparison")
        return
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create the enhanced comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Accuracy Improvement
    colors1 = ['#2E8B57' if x > 0 else '#DC143C' for x in comparison_df['accuracy_improvement']]
    bars1 = ax1.bar(range(len(comparison_df)), comparison_df['accuracy_improvement'], 
                    color=colors1, alpha=0.7)
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax1.set_ylabel('Accuracy Improvement (CoT - ZS)', fontweight='normal')
    ax1.set_title('Accuracy: CoT vs ZS', fontweight='normal')
    ax1.set_xticks(range(0, len(comparison_df), max(1, len(comparison_df)//10)))
    apply_clean_style(ax1)
    
    # Plot 2: ECE Improvement (lower ECE is better, so positive improvement is good)
    colors2 = ['#2E8B57' if x > 0 else '#DC143C' for x in comparison_df['ece_improvement']]
    bars2 = ax2.bar(range(len(comparison_df)), comparison_df['ece_improvement'], 
                    color=colors2, alpha=0.7)
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax2.set_ylabel('Calibration Improvement (ZS_ECE - CoT_ECE)', fontweight='normal')
    ax2.set_title('Calibration Quality: CoT vs ZS', fontweight='normal')
    ax2.set_xticks(range(0, len(comparison_df), max(1, len(comparison_df)//10)))
    apply_clean_style(ax2)
    
    # Plot 3: Brier Score Improvement
    colors3 = ['#2E8B57' if x > 0 else '#DC143C' for x in comparison_df['brier_improvement']]
    bars3 = ax3.bar(range(len(comparison_df)), comparison_df['brier_improvement'], 
                    color=colors3, alpha=0.7)
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax3.set_ylabel('Brier Score Improvement (ZS - CoT)', fontweight='normal')
    ax3.set_title('Overall Performance: CoT vs ZS', fontweight='normal')
    ax3.set_xlabel('Model-Dataset Combinations', fontweight='normal')
    ax3.set_xticks(range(0, len(comparison_df), max(1, len(comparison_df)//10)))
    apply_clean_style(ax3)
    
    # Plot 4: Scatter plot of improvements
    ax4.scatter(comparison_df['accuracy_improvement'], comparison_df['ece_improvement'], 
               alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    # Add quadrant lines
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    ax4.axvline(x=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Label quadrants
    xlim = ax4.get_xlim()
    ylim = ax4.get_ylim()
    ax4.text(xlim[1]*0.7, ylim[1]*0.7, 'Better\nAccuracy &\nCalibration', 
             ha='center', va='center', fontweight='bold', 
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    ax4.text(xlim[0]*0.7, ylim[0]*0.7, 'Worse\nAccuracy &\nCalibration', 
             ha='center', va='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
    
    ax4.set_xlabel('Accuracy Improvement (CoT - ZS)', fontweight='normal')
    ax4.set_ylabel('Calibration Improvement (ZS_ECE - CoT_ECE)', fontweight='normal')
    ax4.set_title('CoT vs ZS: Joint Improvement', fontweight='normal')
    apply_clean_style(ax4)
    
    fig.suptitle(f'Enhanced Method Comparison: {EXP_TYPE_MAPPING["cot_exp"]} vs {EXP_TYPE_MAPPING["zs_exp"]}', 
                 fontweight='bold', fontsize=16)
    plt.tight_layout()
    
    fig.savefig(os.path.join(save_dir, 'enhanced_cot_vs_zs_comparison.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(save_dir, 'enhanced_cot_vs_zs_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # Save the comparison data with enhanced metrics
    comparison_df.to_csv(os.path.join(save_dir, 'enhanced_cot_vs_zs_comparison_data.csv'), index=False)
    
    # Print enhanced summary statistics
    print(f"📊 Enhanced CoT vs ZS Comparison Summary:")
    print(f"   • Models analyzed: {len(comparison_df)}")
    print(f"   • Average accuracy improvement: {comparison_df['accuracy_improvement'].mean():.3f}")
    print(f"   • Average calibration improvement (ECE): {comparison_df['ece_improvement'].mean():.3f}")
    print(f"   • Average Brier score improvement: {comparison_df['brier_improvement'].mean():.3f}")
    print(f"   • Models improved by CoT (accuracy): {(comparison_df['accuracy_improvement'] > 0).sum()}/{len(comparison_df)}")
    print(f"   • Models improved by CoT (calibration): {(comparison_df['ece_improvement'] > 0).sum()}/{len(comparison_df)}")


def create_summary_plots(all_data: Dict, save_dir: str = "summary_plots"):
    """Create enhanced summary plots with proper calibration metrics and alpha shapes"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Create enhanced summary dataframe with calibration metrics
    summary_df = create_enhanced_summary_dataframe(all_data)
    
    if len(summary_df) == 0:
        print("❌ No data available for summary plots")
        return
    
    # Create separate enhanced plots for CoT and ZS experiments
    exp_types = summary_df['exp_type'].unique()
    
    print(f"Creating enhanced summary plots with calibration metrics:")
    
    for exp_type in exp_types:
        exp_name = EXP_TYPE_MAPPING[exp_type]
        exp_short = "cot" if exp_type == "cot_exp" else "zs"
        print(f"   • {exp_name} performance landscape with ECE")
        
        # Create performance landscape with alpha shapes
        create_performance_landscape_with_alphashapes(summary_df, exp_type, save_dir)
    
    # Create cross-dataset consistency analysis
    print(f"   • Cross-dataset consistency profiles")
    create_cross_dataset_consistency(summary_df, save_dir)
    
    # Create enhanced CoT vs ZS comparison
    print(f"   • Enhanced CoT vs ZS comparison with calibration metrics")
    create_calibration_metrics_comparison(summary_df, save_dir)
    
    # Create per-dataset summary plots (updated with ECE)
    print(f"   • Per-dataset calibration summaries")
    create_per_dataset_summary(summary_df, save_dir)
    
    # Save enhanced summary statistics
    summary_df.to_csv(os.path.join(save_dir, 'enhanced_summary_statistics.csv'), index=False)
    
    # Also save separate CSV files for each experiment type
    for exp_type in exp_types:
        exp_data = summary_df[summary_df['exp_type'] == exp_type]
        if len(exp_data) > 0:
            exp_short = "cot" if exp_type == "cot_exp" else "zs"
            exp_data.to_csv(os.path.join(save_dir, f'enhanced_summary_statistics_{exp_short}.csv'), index=False)
    
    print(f"📊 Enhanced summary plots created:")
    for exp_type in exp_types:
        exp_short = "cot" if exp_type == "cot_exp" else "zs"
        exp_name = EXP_TYPE_MAPPING[exp_type]
        print(f"   • {exp_name} ({exp_short.upper()}):")
        print(f"     - {exp_short}_performance_landscape.pdf/png (Accuracy vs ECE with alpha shapes)")
    print(f"   • Cross-Analysis:")
    print(f"     - cross_dataset_consistency.pdf/png")
    print(f"     - enhanced_cot_vs_zs_comparison.pdf/png")
    print(f"   • Per-Dataset Analysis:")
    print(f"     - per_dataset/ (individual calibration plots for each dataset)")
    print(f"   • Data: enhanced_summary_statistics.csv (with ECE, Brier score, overconfidence rate)")


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
            # Use MODEL_NAME_MAPPING for consistent display names
            model_display = MODEL_NAME_MAPPING.get(model, model)
            
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
            
            # Add model name as annotation using MODEL_NAME_MAPPING
            model_display = MODEL_NAME_MAPPING.get(row['model'], row['model'])
            if len(model_display) > 15:
                model_display = model_display[:12] + '...'
            
            # Add experiment type to annotation
            exp_label = exp_names.get(row['exp_type'], row['exp_type'])
            annotation = f"{model_display} ({exp_label})"
            
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
            dataset_data.to_csv(os.path.join(dataset_dir, f'enhanced_summary_statistics_{dataset.lower()}.csv'), index=False)
    
    print(f"📊 Created per-dataset summary plots:")
    for dataset in datasets:
        print(f"   • {dataset.upper()}:")
        print(f"     - calibration_summary_{dataset.lower()}.pdf/png")
        print(f"     - calibration_summary_{dataset.lower()}_annotated.pdf/png")
        print(f"     - enhanced_summary_statistics_{dataset.lower()}.csv") 