#!/usr/bin/env python3
"""
LIF Gradient Analysis Script
Iteration 3: Improved hierarchical analysis with proper data aggregation.

This script extracts all gradient metrics and creates clear, hierarchical visualizations
showing trial-level comparisons with proper aggregation across runs and batches.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
import sys

def extract_gradient_data(db_path):
    """Extract all gradient data using DataFrameExtractor."""
    
    print(f"Extracting gradient data from: {db_path}")
    
    if not os.path.exists(db_path):
        print(f"ERROR: Database file not found: {db_path}")
        return None
    
    try:
        # Import experiment manager modules
        sys.path.append(r'C:\Users\roymi\projects\experiment_manager')
        from experiment_manager.results.extractors.dataframe_extractor import DataFrameExtractor
        from experiment_manager.results.sources.db_datasource import DBDataSource
        
        # Create data source
        print("Creating DBDataSource...")
        datasource = DBDataSource(db_path)
        
        # Create extractor with batch granularity to get gradient data
        print("Creating DataFrameExtractor with batch granularity...")
        extractor = DataFrameExtractor(granularity=['batch'], include_per_label=False)
        
        # Extract data
        print("Calling extractor.extract(datasource)...")
        df = extractor.extract(datasource)
        
        print(f"✓ Extracted {len(df)} gradient records")
        
        if len(df) > 0:
            print(f"  - Columns: {df.columns.tolist()}")
            print(f"  - Gradient metrics: {df['metric'].unique()}")
            print(f"  - Trials: {len(df['trial_name'].unique())}")
            print(f"  - Runs per trial: {df.groupby('trial_name')['trial_run_id'].nunique().to_dict()}")
            print(f"  - Epochs range: {df['epoch'].min()} to {df['epoch'].max()}")
        
        return df
        
    except Exception as e:
        print(f"ERROR extracting data: {e}")
        import traceback
        traceback.print_exc()
        return None

def aggregate_data_hierarchically(df):
    """Aggregate data hierarchically: batch → epoch → run → trial."""
    
    print("Aggregating data hierarchically...")
    
    # Step 1: Aggregate batches to epoch level (mean/std per epoch per run)
    print("  Step 1: Aggregating batches to epoch level...")
    epoch_data = []
    
    for trial_name in df['trial_name'].unique():
        trial_data = df[df['trial_name'] == trial_name]
        
        for run_id in trial_data['trial_run_id'].unique():
            run_data = trial_data[trial_data['trial_run_id'] == run_id]
            
            for metric in run_data['metric'].unique():
                metric_data = run_data[run_data['metric'] == metric]
                
                # Aggregate batches within each epoch
                epoch_agg = metric_data.groupby('epoch').agg({
                    'value': ['mean', 'std', 'min', 'max', 'count']
                }).reset_index()
                
                # Flatten column names
                epoch_agg.columns = ['epoch', 'mean', 'std', 'min', 'max', 'batch_count']
                
                # Add metadata
                epoch_agg['trial_name'] = trial_name
                epoch_agg['trial_run_id'] = run_id
                epoch_agg['metric'] = metric
                
                epoch_data.append(epoch_agg)
    
    epoch_df = pd.concat(epoch_data, ignore_index=True)
    print(f"    ✓ Created {len(epoch_df)} epoch-level records")
    
    # Step 2: Aggregate runs to trial level (mean/std across runs per epoch)
    print("  Step 2: Aggregating runs to trial level...")
    trial_data = []
    
    for trial_name in epoch_df['trial_name'].unique():
        trial_epoch_data = epoch_df[epoch_df['trial_name'] == trial_name]
        
        for metric in trial_epoch_data['metric'].unique():
            metric_data = trial_epoch_data[trial_epoch_data['metric'] == metric]
            
            # Aggregate runs within each epoch
            trial_agg = metric_data.groupby('epoch').agg({
                'mean': ['mean', 'std'],  # Mean and std of epoch means across runs
                'std': 'mean',            # Mean of epoch stds across runs
                'min': 'min',             # Overall min across runs
                'max': 'max',             # Overall max across runs
                'batch_count': 'sum'      # Total batches across runs
            }).reset_index()
            
            # Flatten column names
            trial_agg.columns = ['epoch', 'mean_mean', 'mean_std', 'std_mean', 'min', 'max', 'total_batches']
            
            # Add metadata
            trial_agg['trial_name'] = trial_name
            trial_agg['metric'] = metric
            
            trial_data.append(trial_agg)
    
    trial_df = pd.concat(trial_data, ignore_index=True)
    print(f"    ✓ Created {len(trial_df)} trial-level records")
    
    return epoch_df, trial_df

def create_trial_comparison_plots(trial_df, output_dir="./"):
    """Create clear trial-level comparison plots."""
    
    print("Creating trial-level comparison plots...")
    
    # Set up plotting style
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': (16, 12),
        'font.size': 10,
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    # Get unique gradient metrics and trials
    gradient_metrics = trial_df['metric'].unique()
    trials = trial_df['trial_name'].unique()
    
    # Create color and line style mapping
    lr_colors = {
        'lr_0.01': 'blue',
        'lr_0.1': 'green', 
        'lr_1': 'orange',
        'lr_10': 'red'
    }
    
    len_styles = {
        'len_50': '-',
        'len_75': '--',
        'len_100': ':'
    }
    
    # Create main comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(gradient_metrics):
        if i >= len(axes):
            break
            
        metric_data = trial_df[trial_df['metric'] == metric]
        
        # Plot each trial with proper styling
        for trial_name in trials:
            trial_data = metric_data[metric_data['trial_name'] == trial_name]
            
            if len(trial_data) > 0:
                # Determine color and line style
                color = 'gray'  # default
                linestyle = '-'
                
                for lr_key, lr_color in lr_colors.items():
                    if lr_key in trial_name:
                        color = lr_color
                        break
                
                for len_key, len_style in len_styles.items():
                    if len_key in trial_name:
                        linestyle = len_style
                        break
                
                # Plot mean with error bands
                axes[i].plot(trial_data['epoch'], trial_data['mean_mean'],
                           linestyle=linestyle, color=color, linewidth=2,
                           label=trial_name, alpha=0.8)
                
                # Add error bands (std across runs)
                axes[i].fill_between(trial_data['epoch'],
                                   trial_data['mean_mean'] - trial_data['mean_std'],
                                   trial_data['mean_mean'] + trial_data['mean_std'],
                                   alpha=0.2, color=color)
        
        axes[i].set_title(f'{metric}\nTrial Comparison (Mean ± Std across runs)', fontsize=12)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(f'{metric} Value')
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(gradient_metrics), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, "gradient_analysis_trial_comparison.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()

def create_run_consistency_plots(epoch_df, output_dir="./"):
    """Create run consistency analysis plots."""
    
    print("Creating run consistency analysis plots...")
    
    # Select a few representative trials for detailed analysis
    trials = epoch_df['trial_name'].unique()
    selected_trials = trials[:4]  # Show first 4 trials
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, trial_name in enumerate(selected_trials):
        if i >= len(axes):
            break
            
        trial_data = epoch_df[epoch_df['trial_name'] == trial_name]
        
        # Plot all runs for this trial (focus on gradient_l2_norm)
        metric_data = trial_data[trial_data['metric'] == 'gradient_l2_norm']
        
        runs = metric_data['trial_run_id'].unique()
        colors = plt.cm.tab10(range(len(runs)))
        
        for j, run_id in enumerate(runs):
            run_data = metric_data[metric_data['trial_run_id'] == run_id]
            
            axes[i].plot(run_data['epoch'], run_data['mean'],
                        alpha=0.7, linewidth=1.5, color=colors[j],
                        label=f'Run {j+1}')
        
        axes[i].set_title(f'{trial_name}\nRun Consistency (gradient_l2_norm)', fontsize=12)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('gradient_l2_norm Value')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(selected_trials), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, "gradient_analysis_run_consistency.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()

def create_statistical_summary_plots(trial_df, output_dir="./"):
    """Create statistical summary plots."""
    
    print("Creating statistical summary plots...")
    
    # Set up plotting style
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': (16, 12),
        'font.size': 10,
        'axes.grid': True,
        'grid.alpha': 0.3
    })
    
    # Create box plots for each metric
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    gradient_metrics = trial_df['metric'].unique()
    
    for i, metric in enumerate(gradient_metrics):
        if i >= len(axes):
            break
            
        metric_data = trial_df[trial_df['metric'] == metric]
        
        # Prepare data for box plot (final epoch values across trials)
        final_epoch_data = metric_data[metric_data['epoch'] == metric_data['epoch'].max()]
        
        # Create box plot
        trial_names = final_epoch_data['trial_name'].tolist()
        values = final_epoch_data['mean_mean'].tolist()
        
        bp = axes[i].boxplot([values], labels=[f'{metric}\nFinal Epoch'], patch_artist=True)
        
        # Color boxes by learning rate
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        
        axes[i].set_title(f'{metric}\nFinal Epoch Distribution Across Trials', fontsize=12)
        axes[i].set_ylabel(f'{metric} Value')
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(len(gradient_metrics), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, "gradient_analysis_statistical_summary.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()
    
    # Create heatmap showing gradient evolution
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(gradient_metrics):
        if i >= len(axes):
            break
            
        metric_data = trial_df[trial_df['metric'] == metric]
        
        # Create pivot table for heatmap
        pivot_data = metric_data.pivot(index='epoch', columns='trial_name', values='mean_mean')
        
        # Create heatmap
        im = axes[i].imshow(pivot_data.T, aspect='auto', cmap='viridis')
        axes[i].set_title(f'{metric}\nGradient Evolution Heatmap', fontsize=12)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Trial')
        axes[i].set_yticks(range(len(pivot_data.columns)))
        axes[i].set_yticklabels(pivot_data.columns, fontsize=8)
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i])
    
    # Hide unused subplots
    for i in range(len(gradient_metrics), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, "gradient_analysis_heatmaps.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")
    plt.show()

def analyze_gradient_behavior(trial_df):
    """Provide comprehensive statistical analysis of gradient behavior."""
    
    if len(trial_df) == 0:
        print("No data to analyze")
        return
    
    print("\n=== Comprehensive Gradient Behavior Analysis ===")
    
    # Analyze each gradient metric
    for metric in trial_df['metric'].unique():
        metric_data = trial_df[trial_df['metric'] == metric]
        
        print(f"\n{metric}:")
        print(f"  Overall range: {metric_data['mean_mean'].min():.6f} to {metric_data['mean_mean'].max():.6f}")
        print(f"  Overall mean: {metric_data['mean_mean'].mean():.6f}")
        print(f"  Overall std: {metric_data['mean_mean'].std():.6f}")
        
        # Analysis by trial
        print(f"  By trial (final epoch):")
        final_epoch_data = metric_data[metric_data['epoch'] == metric_data['epoch'].max()]
        
        for _, row in final_epoch_data.iterrows():
            trial_name = row['trial_name']
            mean_val = row['mean_mean']
            std_val = row['mean_std']
            run_std = row['std_mean']
            
            print(f"    {trial_name}: mean={mean_val:.6f}, run_std={std_val:.6f}, epoch_std={run_std:.6f}")
        
        # Check for potential issues
        if metric_data['mean_mean'].min() < 1e-6:
            print(f"  ⚠️  WARNING: Very small gradients detected (min: {metric_data['mean_mean'].min():.2e}) - possible vanishing gradients")
        
        if metric_data['mean_mean'].max() > 10:
            print(f"  ⚠️  WARNING: Large gradients detected (max: {metric_data['mean_mean'].max():.2e}) - possible exploding gradients")
        
        # Check for high run-to-run variability
        high_variability = final_epoch_data[final_epoch_data['mean_std'] > final_epoch_data['mean_mean'] * 0.5]
        if len(high_variability) > 0:
            print(f"  ⚠️  WARNING: High run-to-run variability detected in {len(high_variability)} trials")

def main():
    """Main analysis function."""
    
    print("=== LIF Gradient Analysis - Iteration 3 (Hierarchical) ===")
    
    # Configuration
    db_path = r'outputs\sequential_workspace\artifacts\experiment.db'
    
    # Step 1: Extract gradient data
    print("\n1. Extracting gradient data...")
    df = extract_gradient_data(db_path)
    
    if df is None or len(df) == 0:
        print("✗ No gradient data found or extraction failed!")
        return
    
    # Step 2: Aggregate data hierarchically
    print("\n2. Aggregating data hierarchically...")
    epoch_df, trial_df = aggregate_data_hierarchically(df)
    
    # Step 3: Statistical analysis
    print("\n3. Analyzing gradient behavior...")
    analyze_gradient_behavior(trial_df)
    
    # Step 4: Create hierarchical plots
    print("\n4. Creating hierarchical visualization plots...")
    create_trial_comparison_plots(trial_df)
    create_run_consistency_plots(epoch_df)
    create_statistical_summary_plots(trial_df)
    
    print("\n✓ Hierarchical gradient analysis completed successfully!")
    print("Check the generated PNG files for clear, hierarchical visualizations.")

if __name__ == "__main__":
    main()