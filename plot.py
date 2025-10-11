#!/usr/bin/env python3
"""
LIF Comprehensive Analysis Script
Enhanced version with all metrics and per-trial analysis.

This script extracts all metrics from LIF experiments and creates comprehensive
per-trial analysis with organized directory structure and detailed visualizations.
"""

import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from scipy import stats
from pathlib import Path

def extract_validation_accuracy_from_batch_metric(db_path):
    """Extract validation accuracy from BATCH_METRIC table."""
    
    print("Extracting validation accuracy from BATCH_METRIC table...")
    
    try:
        conn = sqlite3.connect(db_path)
        
        # Get validation accuracy data from BATCH_METRIC and METRIC tables
        val_acc_data = pd.read_sql_query("""
            SELECT 
                bm.batch_idx,
                bm.epoch_idx,
                bm.trial_run_id,
                m.id as metric_id,
                m.type,
                m.total_val as value
            FROM BATCH_METRIC bm
            JOIN METRIC m ON bm.metric_id = m.id
            WHERE m.type = 'val_acc'
            ORDER BY bm.trial_run_id, bm.epoch_idx
        """, conn)
        
        if len(val_acc_data) == 0:
            print("  No validation accuracy found in BATCH_METRIC")
            conn.close()
            return pd.DataFrame()
        
        print(f"  Found {len(val_acc_data)} validation accuracy records")
        
        # Get trial information
        trial_info = pd.read_sql_query("""
            SELECT 
                tr.id as trial_run_id,
                t.name as trial_name
            FROM TRIAL_RUN tr
            JOIN TRIAL t ON tr.trial_id = t.id
        """, conn)
        
        # Merge trial information
        val_acc_data = val_acc_data.merge(trial_info, on='trial_run_id', how='left')
        
        # Add required columns for compatibility
        val_acc_data['experiment_id'] = 1  # Default value
        val_acc_data['trial_id'] = 1  # Default value
        val_acc_data['trial_run_status'] = 'FINISHED'  # Default value
        val_acc_data['granularity'] = 'epoch'
        val_acc_data['batch'] = None
        val_acc_data['is_custom'] = False
        val_acc_data['timestamp'] = pd.Timestamp.now()  # Default timestamp
        val_acc_data['experiment_name'] = 'sequential_workspace'  # Default value
        
        # Rename columns to match expected format
        val_acc_data = val_acc_data.rename(columns={'epoch_idx': 'epoch', 'type': 'metric'})
        
        # Select only the columns we need
        val_acc_data = val_acc_data[[
            'experiment_id', 'experiment_name', 'trial_id', 'trial_name', 
            'trial_run_id', 'trial_run_status', 'granularity', 'epoch', 
            'batch', 'metric', 'value', 'is_custom', 'timestamp'
        ]]
        
        print(f"  [OK] Created {len(val_acc_data)} validation accuracy records")
        print(f"  - Sample data:")
        print(val_acc_data.head())
        
        conn.close()
        return val_acc_data
        
    except Exception as e:
        print(f"  ERROR extracting validation accuracy: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def extract_all_metrics(db_path):
    """Extract all available metrics using DataFrameExtractor."""
    
    print(f"Extracting all metrics from: {db_path}")
    
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
        
        # Extract batch-level data (for gradient analysis)
        print("Extracting batch-level data...")
        batch_extractor = DataFrameExtractor(granularity=['batch'], include_per_label=False)
        batch_df = batch_extractor.extract(datasource)
        
        # Extract epoch-level data (for training metrics)
        print("Extracting epoch-level data...")
        epoch_extractor = DataFrameExtractor(granularity=['epoch'], include_per_label=False)
        epoch_df = epoch_extractor.extract(datasource)
        
        # Extract results-level data (for final metrics)
        print("Extracting results-level data...")
        results_extractor = DataFrameExtractor(granularity=['results'], include_per_label=False)
        results_df = results_extractor.extract(datasource)
        
        print(f"[OK] Extracted data:")
        print(f"  - Batch records: {len(batch_df)}")
        print(f"  - Epoch records: {len(epoch_df)}")
        print(f"  - Results records: {len(results_df)}")
        
        if len(batch_df) > 0:
            print(f"  - Batch columns: {batch_df.columns.tolist()}")
            print(f"  - Batch metrics: {batch_df['metric'].unique()}")
        if len(epoch_df) > 0:
            print(f"  - Epoch columns: {epoch_df.columns.tolist()}")
            print(f"  - Epoch metrics: {epoch_df['metric'].unique()}")
        if len(results_df) > 0:
            print(f"  - Results columns: {results_df.columns.tolist()}")
            print(f"  - Results metrics: {results_df['metric'].unique()}")
        
        # If epoch data is empty, create it from batch data
        if len(epoch_df) == 0 and len(batch_df) > 0:
            print("Creating epoch-level data from batch data...")
            epoch_df = create_epoch_data_from_batch(batch_df)
            print(f"  [OK] Created {len(epoch_df)} epoch-level records from batch data")
            print(f"  - Created epoch metrics: {epoch_df['metric'].unique()}")
        
        # Extract validation accuracy from BATCH_METRIC table
        val_acc_df = extract_validation_accuracy_from_batch_metric(db_path)
        if len(val_acc_df) > 0:
            # Combine with existing epoch data
            epoch_df = pd.concat([epoch_df, val_acc_df], ignore_index=True)
            print(f"  [OK] Added validation accuracy to epoch data")
            print(f"  - Updated epoch metrics: {epoch_df['metric'].unique()}")
        
        return batch_df, epoch_df, results_df
        
    except Exception as e:
        print(f"ERROR extracting data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def create_epoch_data_from_batch(batch_df):
    """Create epoch-level data by aggregating batch data."""
    
    print("  Aggregating batch data to epoch level...")
    epoch_data = []
    
    for trial_name in batch_df['trial_name'].unique():
        trial_data = batch_df[batch_df['trial_name'] == trial_name]
        
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
                
                # Add other required columns to match expected format
                epoch_agg['experiment_id'] = metric_data['experiment_id'].iloc[0]
                epoch_agg['experiment_name'] = metric_data['experiment_name'].iloc[0]
                epoch_agg['trial_id'] = metric_data['trial_id'].iloc[0]
                epoch_agg['trial_run_status'] = metric_data['trial_run_status'].iloc[0]
                epoch_agg['granularity'] = 'epoch'
                epoch_agg['batch'] = None
                epoch_agg['is_custom'] = metric_data['is_custom'].iloc[0]
                epoch_agg['timestamp'] = metric_data['timestamp'].iloc[0]
                
                # Use mean value as the main value for epoch-level data
                epoch_agg['value'] = epoch_agg['mean']
                
                epoch_data.append(epoch_agg)
    
    if len(epoch_data) > 0:
        epoch_df = pd.concat(epoch_data, ignore_index=True)
        return epoch_df
    else:
        return pd.DataFrame()

def create_trial_analysis_structure(base_dir, trial_name):
    """Create organized directory structure for each trial."""
    
    # Create main analysis directory
    analysis_dir = Path(base_dir) / "analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # Create trial-specific directory
    trial_dir = analysis_dir / trial_name
    trial_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    subdirs = ["training_progress", "gradient_analysis", "convergence", "performance"]
    for subdir in subdirs:
        (trial_dir / subdir).mkdir(exist_ok=True)
    
    return trial_dir

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
    print(f"    [OK] Created {len(epoch_df)} epoch-level records")
    
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
    print(f"    [OK] Created {len(trial_df)} trial-level records")
    
    return epoch_df, trial_df

class PerTrialAnalyzer:
    """Analyzer for individual trial analysis."""
    
    def __init__(self, trial_name, batch_df, epoch_df, results_df, output_dir):
        self.trial_name = trial_name
        self.batch_df = batch_df[batch_df['trial_name'] == trial_name] if batch_df is not None else pd.DataFrame()
        
        # Handle epoch_df which might not have trial_name column
        if epoch_df is not None and len(epoch_df) > 0:
            if 'trial_name' in epoch_df.columns:
                self.epoch_df = epoch_df[epoch_df['trial_name'] == trial_name]
            else:
                # If no trial_name column, filter by trial_id
                trial_id = batch_df[batch_df['trial_name'] == trial_name]['trial_id'].iloc[0] if len(batch_df) > 0 else None
                if trial_id is not None:
                    self.epoch_df = epoch_df[epoch_df['trial_id'] == trial_id]
                else:
                    self.epoch_df = pd.DataFrame()
        else:
            self.epoch_df = pd.DataFrame()
        
        # Handle results_df which might not have trial_name column
        if results_df is not None and len(results_df) > 0:
            if 'trial_name' in results_df.columns:
                self.results_df = results_df[results_df['trial_name'] == trial_name]
            else:
                # If no trial_name column, filter by trial_id
                trial_id = batch_df[batch_df['trial_name'] == trial_name]['trial_id'].iloc[0] if len(batch_df) > 0 else None
                if trial_id is not None:
                    self.results_df = results_df[results_df['trial_id'] == trial_id]
                else:
                    self.results_df = pd.DataFrame()
        else:
            self.results_df = pd.DataFrame()
        
        self.output_dir = Path(output_dir)
        
        # Set up plotting style
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 10,
            'axes.grid': True,
            'grid.alpha': 0.3
        })
    
    def create_training_progress_plots(self):
        """Create training progress plots."""
        
        print(f"  Creating training progress plots for {self.trial_name}...")
        
        # Debug: Show available metrics
        if len(self.epoch_df) > 0:
            print(f"    Available epoch metrics: {self.epoch_df['metric'].unique()}")
        
        # Training loss plot
        if len(self.epoch_df) > 0:
            loss_data = self.epoch_df[self.epoch_df['metric'] == 'train_loss']
            if len(loss_data) > 0:
                self._plot_training_metric(loss_data, 'Training Loss', 'training_progress/loss_curves.png')
            
            # Validation loss plot
            val_loss_data = self.epoch_df[self.epoch_df['metric'] == 'val_loss']
            if len(val_loss_data) > 0:
                self._plot_training_metric(val_loss_data, 'Validation Loss', 'training_progress/val_loss_curves.png')
            
            # Validation accuracy plot
            val_acc_data = self.epoch_df[self.epoch_df['metric'] == 'val_acc']
            if len(val_acc_data) > 0:
                self._plot_training_metric(val_acc_data, 'Validation Accuracy', 'training_progress/accuracy_curves.png')
            
            # Learning rate plot (if available)
            lr_data = self.epoch_df[self.epoch_df['metric'] == 'learning_rate']
            if len(lr_data) > 0:
                self._plot_training_metric(lr_data, 'Learning Rate', 'training_progress/learning_rate.png')
            
            # Create comprehensive training dashboard
            self._create_training_dashboard()
    
    def _create_training_dashboard(self):
        """Create a comprehensive training dashboard with all available metrics."""
        
        # Get all available metrics
        available_metrics = self.epoch_df['metric'].unique()
        
        # Create subplots based on available metrics
        n_metrics = len(available_metrics)
        if n_metrics == 0:
            return
        
        # Create a grid layout
        cols = min(3, n_metrics)
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics):
            if i >= len(axes):
                break
                
            metric_data = self.epoch_df[self.epoch_df['metric'] == metric]
            
            # Plot each run
            runs = metric_data['trial_run_id'].unique()
            colors = plt.cm.tab10(range(len(runs)))
            
            for j, run_id in enumerate(runs):
                run_data = metric_data[metric_data['trial_run_id'] == run_id]
                axes[i].plot(run_data['epoch'], run_data['value'],
                           alpha=0.7, linewidth=2, color=colors[j],
                           label=f'Run {j+1}')
            
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_progress/training_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_gradient_analysis_plots(self):
        """Create gradient analysis plots."""
        
        print(f"  Creating gradient analysis plots for {self.trial_name}...")
        
        if len(self.batch_df) == 0:
            return
        
        # Get gradient metrics
        gradient_metrics = [m for m in self.batch_df['metric'].unique() if 'gradient' in m]
        
        if len(gradient_metrics) == 0:
            return
        
        # Create gradient evolution plot
        self._plot_gradient_evolution(gradient_metrics)
        
        # Create gradient distributions plot
        self._plot_gradient_distributions(gradient_metrics)
        
        # Create anomaly detection plot
        self._plot_gradient_anomalies(gradient_metrics)
    
    def create_convergence_plots(self):
        """Create convergence analysis plots."""
        
        print(f"  Creating convergence plots for {self.trial_name}...")
        
        if len(self.epoch_df) == 0:
            return
        
        # Train vs validation loss
        train_loss = self.epoch_df[self.epoch_df['metric'] == 'train_loss']
        val_loss = self.epoch_df[self.epoch_df['metric'] == 'val_loss']
        
        if len(train_loss) > 0 and len(val_loss) > 0:
            self._plot_train_vs_val(train_loss, val_loss)
        elif len(train_loss) > 0:
            # If only training loss is available, create a training loss only plot
            self._plot_training_loss_only(train_loss)
    
    def create_performance_summary(self):
        """Create performance summary plots."""
        
        print(f"  Creating performance summary for {self.trial_name}...")
        
        # Final metrics comparison
        if len(self.results_df) > 0:
            self._plot_final_metrics()
        
        # Run comparison
        if len(self.epoch_df) > 0:
            self._plot_run_comparison()
        
        # Create comprehensive performance dashboard
        self._create_performance_dashboard()
    
    def generate_summary_report(self):
        """Generate text summary report."""
        
        print(f"  Generating summary report for {self.trial_name}...")
        
        report_path = self.output_dir / "summary_report.txt"
        
        with open(report_path, 'w') as f:
            f.write(f"Trial Analysis Report: {self.trial_name}\n")
            f.write("=" * 50 + "\n\n")
            
            # Training summary
            f.write("Training Summary:\n")
            f.write("-" * 20 + "\n")
            
            if len(self.epoch_df) > 0:
                train_loss = self.epoch_df[self.epoch_df['metric'] == 'train_loss']
                if len(train_loss) > 0:
                    # Get final training loss
                    final_loss = train_loss[train_loss['epoch'] == train_loss['epoch'].max()]['value'].mean()
                    f.write(f"- Final training loss: {final_loss:.6f}\n")
                    
                    # Get training loss statistics
                    all_train_losses = train_loss['value'].values
                    f.write(f"- Training loss range: {all_train_losses.min():.6f} to {all_train_losses.max():.6f}\n")
                    f.write(f"- Training loss mean: {all_train_losses.mean():.6f}\n")
                    f.write(f"- Training loss std: {all_train_losses.std():.6f}\n")
                    
                    # Check for convergence
                    runs = train_loss['trial_run_id'].unique()
                    convergence_info = []
                    for run_id in runs:
                        run_data = train_loss[train_loss['trial_run_id'] == run_id]
                        if len(run_data) >= 10:  # Need at least 10 epochs to check convergence
                            early_loss = run_data[run_data['epoch'] <= 10]['value'].mean()
                            late_loss = run_data[run_data['epoch'] >= run_data['epoch'].max() - 9]['value'].mean()
                            improvement = early_loss - late_loss
                            convergence_info.append(improvement)
                    
                    if convergence_info:
                        avg_improvement = np.mean(convergence_info)
                        f.write(f"- Average loss improvement: {avg_improvement:.6f}\n")
                        if avg_improvement > 0.01:
                            f.write("- [OK] Training shows good convergence\n")
                        else:
                            f.write("- [WARNING] Training may not be converging well\n")
            
            # Validation metrics (if available)
            if len(self.epoch_df) > 0:
                val_loss = self.epoch_df[self.epoch_df['metric'] == 'val_loss']
                val_acc = self.epoch_df[self.epoch_df['metric'] == 'val_acc']
                
                if len(val_loss) > 0:
                    final_val_loss = val_loss[val_loss['epoch'] == val_loss['epoch'].max()]['value'].mean()
                    f.write(f"- Final validation loss: {final_val_loss:.6f}\n")
                
                if len(val_acc) > 0:
                    final_val_acc = val_acc[val_acc['epoch'] == val_acc['epoch'].max()]['value'].mean()
                    f.write(f"- Final validation accuracy: {final_val_acc:.2f}%\n")
            
            # Gradient summary
            f.write("\nGradient Health:\n")
            f.write("-" * 20 + "\n")
            
            if len(self.batch_df) > 0:
                gradient_metrics = [m for m in self.batch_df['metric'].unique() if 'gradient' in m]
                for metric in gradient_metrics:
                    metric_data = self.batch_df[self.batch_df['metric'] == metric]
                    mean_val = metric_data['value'].mean()
                    std_val = metric_data['value'].std()
                    min_val = metric_data['value'].min()
                    max_val = metric_data['value'].max()
                    
                    f.write(f"- {metric}:\n")
                    f.write(f"  Mean: {mean_val:.6f}, Std: {std_val:.6f}\n")
                    f.write(f"  Range: {min_val:.6f} to {max_val:.6f}\n")
                    
                    # Check for gradient issues
                    if metric == 'gradient_l2_norm':
                        if mean_val < 0.01:
                            f.write("  [WARNING] Very small gradients (vanishing gradients)\n")
                        elif mean_val > 10:
                            f.write("  [WARNING] Very large gradients (exploding gradients)\n")
                        else:
                            f.write("  [OK] Gradient magnitude appears healthy\n")
                    
                    if metric == 'gradient_std' and std_val > mean_val * 2:
                        f.write("  [WARNING] High gradient variability\n")
            
            # Performance summary
            f.write("\nPerformance:\n")
            f.write("-" * 20 + "\n")
            
            if len(self.results_df) > 0:
                for _, row in self.results_df.iterrows():
                    metric_name = row['metric']
                    value = row['value']
                    f.write(f"- {metric_name}: {value:.6f}\n")
                    
                    # Add interpretation for accuracy metrics
                    if 'acc' in metric_name:
                        if value >= 90:
                            f.write("  [OK] Excellent performance\n")
                        elif value >= 80:
                            f.write("  [OK] Good performance\n")
                        elif value >= 70:
                            f.write("  [WARNING] Moderate performance\n")
                        elif value >= 50:
                            f.write("  [WARNING] Poor performance\n")
                        else:
                            f.write("  [ERROR] Very poor performance\n")
                    
                    # Add interpretation for loss metrics
                    if 'loss' in metric_name:
                        if value < 0.1:
                            f.write("  [OK] Very low loss\n")
                        elif value < 0.5:
                            f.write("  [OK] Low loss\n")
                        elif value < 1.0:
                            f.write("  [WARNING] Moderate loss\n")
                        else:
                            f.write("  [ERROR] High loss\n")
            
            # Run consistency analysis
            f.write("\nRun Consistency:\n")
            f.write("-" * 20 + "\n")
            
            if len(self.epoch_df) > 0:
                train_loss = self.epoch_df[self.epoch_df['metric'] == 'train_loss']
                if len(train_loss) > 0:
                    runs = train_loss['trial_run_id'].unique()
                    f.write(f"- Number of runs: {len(runs)}\n")
                    
                    if len(runs) > 1:
                        # Calculate final loss for each run
                        final_losses = []
                        for run_id in runs:
                            run_data = train_loss[train_loss['trial_run_id'] == run_id]
                            final_epoch = run_data['epoch'].max()
                            final_loss = run_data[run_data['epoch'] == final_epoch]['value'].iloc[0]
                            final_losses.append(final_loss)
                        
                        final_losses = np.array(final_losses)
                        f.write(f"- Final loss mean: {final_losses.mean():.6f}\n")
                        f.write(f"- Final loss std: {final_losses.std():.6f}\n")
                        f.write(f"- Final loss CV: {final_losses.std()/final_losses.mean()*100:.2f}%\n")
                        
                        if final_losses.std() < final_losses.mean() * 0.1:
                            f.write("- [OK] High run consistency\n")
                        elif final_losses.std() < final_losses.mean() * 0.3:
                            f.write("- [WARNING] Moderate run consistency\n")
                        else:
                            f.write("- [ERROR] Low run consistency\n")
            
            # Overall assessment
            f.write("\nOverall Assessment:\n")
            f.write("-" * 20 + "\n")
            
            # Determine overall performance
            performance_score = 0
            issues = []
            
            # Check training convergence
            if len(self.epoch_df) > 0:
                train_loss = self.epoch_df[self.epoch_df['metric'] == 'train_loss']
                if len(train_loss) > 0:
                    final_loss = train_loss[train_loss['epoch'] == train_loss['epoch'].max()]['value'].mean()
                    if final_loss < 0.1:
                        performance_score += 2
                    elif final_loss < 0.5:
                        performance_score += 1
                    else:
                        issues.append("High training loss")
            
            # Check test performance
            if len(self.results_df) > 0:
                test_acc = self.results_df[self.results_df['metric'] == 'test_acc']
                if len(test_acc) > 0:
                    acc_value = test_acc['value'].iloc[0]
                    if acc_value >= 80:
                        performance_score += 3
                    elif acc_value >= 60:
                        performance_score += 2
                    elif acc_value >= 40:
                        performance_score += 1
                    else:
                        issues.append("Low test accuracy")
            
            # Check gradient health
            if len(self.batch_df) > 0:
                gradient_l2 = self.batch_df[self.batch_df['metric'] == 'gradient_l2_norm']
                if len(gradient_l2) > 0:
                    l2_mean = gradient_l2['value'].mean()
                    if 0.1 <= l2_mean <= 10:
                        performance_score += 1
                    else:
                        issues.append("Unhealthy gradients")
            
            # Overall rating
            if performance_score >= 5:
                f.write("[EXCELLENT] This configuration performs very well\n")
            elif performance_score >= 3:
                f.write("[GOOD] This configuration performs well\n")
            elif performance_score >= 1:
                f.write("[MODERATE] This configuration has some issues\n")
            else:
                f.write("[POOR] This configuration has significant issues\n")
            
            if issues:
                f.write(f"Main issues: {', '.join(issues)}\n")
            
            f.write(f"Performance score: {performance_score}/6\n")
    
    def _plot_training_metric(self, data, title, filename):
        """Plot training metric with run variability."""
        
        plt.figure(figsize=(12, 8))
        
        # Plot each run
        runs = data['trial_run_id'].unique()
        colors = plt.cm.tab10(range(len(runs)))
        
        for i, run_id in enumerate(runs):
            run_data = data[data['trial_run_id'] == run_id]
            plt.plot(run_data['epoch'], run_data['value'],
                    alpha=0.7, linewidth=2, color=colors[i],
                    label=f'Run {i+1}')
        
        plt.title(f'{title} - {self.trial_name}')
        plt.xlabel('Epoch')
        plt.ylabel(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.output_dir / filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_gradient_evolution(self, gradient_metrics):
        """Plot gradient evolution over epochs."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(gradient_metrics):
            if i >= len(axes):
                break
            
            metric_data = self.batch_df[self.batch_df['metric'] == metric]
            
            # Aggregate by epoch - only include trials that have data for each epoch
            epoch_agg = calculate_epoch_average_with_sample_size(metric_data)
            
            axes[i].plot(epoch_agg['epoch'], epoch_agg['mean'], linewidth=2)
            axes[i].fill_between(epoch_agg['epoch'],
                               epoch_agg['mean'] - epoch_agg['std'],
                               epoch_agg['mean'] + epoch_agg['std'],
                               alpha=0.3)
            
            # Add sample size annotations
            total_trials = len(metric_data['trial_name'].unique())
            for _, row in epoch_agg.iterrows():
                if row['n_trials'] < total_trials:
                    axes[i].annotate(f"n={int(row['n_trials'])}", 
                                   xy=(row['epoch'], row['mean']), 
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, alpha=0.7)
            
            axes[i].set_title(f'{metric} Evolution')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric)
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(gradient_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "gradient_analysis/gradient_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_gradient_distributions(self, gradient_metrics):
        """Plot gradient distributions."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(gradient_metrics):
            if i >= len(axes):
                break
            
            metric_data = self.batch_df[self.batch_df['metric'] == metric]
            
            axes[i].hist(metric_data['value'], bins=50, alpha=0.7, edgecolor='black')
            axes[i].set_title(f'{metric} Distribution')
            axes[i].set_xlabel(metric)
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(gradient_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "gradient_analysis/gradient_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_gradient_anomalies(self, gradient_metrics):
        """Plot gradient anomalies."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(gradient_metrics):
            if i >= len(axes):
                break
            
            metric_data = self.batch_df[self.batch_df['metric'] == metric]
            
            # Detect anomalies using z-score
            z_scores = np.abs(stats.zscore(metric_data['value']))
            anomalies = metric_data[z_scores > 3]
            normal = metric_data[z_scores <= 3]
            
            # Plot normal points
            if len(normal) > 0:
                axes[i].scatter(normal['epoch'], normal['value'],
                              alpha=0.3, s=10, color='blue', label='Normal')
            
            # Plot anomalies
            if len(anomalies) > 0:
                axes[i].scatter(anomalies['epoch'], anomalies['value'],
                              alpha=0.8, s=30, color='red', marker='x', label='Anomalies')
            
            axes[i].set_title(f'{metric} Anomaly Detection')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(gradient_metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "gradient_analysis/anomaly_detection.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_train_vs_val(self, train_loss, val_loss):
        """Plot training vs validation loss."""
        
        plt.figure(figsize=(12, 8))
        
        # Plot training loss
        train_agg = calculate_epoch_average_with_sample_size(train_loss)
        plt.plot(train_agg['epoch'], train_agg['mean'], label='Training Loss', linewidth=2)
        
        # Plot validation loss
        val_agg = calculate_epoch_average_with_sample_size(val_loss)
        plt.plot(val_agg['epoch'], val_agg['mean'], label='Validation Loss', linewidth=2)
        
        plt.title(f'Train vs Validation Loss - {self.trial_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.output_dir / "convergence/train_vs_val_loss.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_training_loss_only(self, train_loss):
        """Plot training loss when validation loss is not available."""
        
        plt.figure(figsize=(12, 8))
        
        # Plot training loss
        train_agg = calculate_epoch_average_with_sample_size(train_loss)
        plt.plot(train_agg['epoch'], train_agg['mean'], label='Training Loss', linewidth=2, color='blue')
        
        plt.title(f'Training Loss - {self.trial_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.output_dir / "convergence/training_loss.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_final_metrics(self):
        """Plot final metrics."""
        
        plt.figure(figsize=(10, 6))
        
        metrics = self.results_df['metric'].tolist()
        values = self.results_df['value'].tolist()
        
        bars = plt.bar(metrics, values)
        plt.title(f'Final Metrics - {self.trial_name}')
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance/final_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_run_comparison(self):
        """Plot run comparison."""
        
        # Focus on training loss for run comparison
        train_loss = self.epoch_df[self.epoch_df['metric'] == 'train_loss']
        
        if len(train_loss) == 0:
            return
        
        plt.figure(figsize=(12, 8))
        
        runs = train_loss['trial_run_id'].unique()
        colors = plt.cm.tab10(range(len(runs)))
        
        for i, run_id in enumerate(runs):
            run_data = train_loss[train_loss['trial_run_id'] == run_id]
            plt.plot(run_data['epoch'], run_data['value'],
                    alpha=0.8, linewidth=2, color=colors[i],
                    label=f'Run {i+1}')
        
        plt.title(f'Run Comparison - Training Loss - {self.trial_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(self.output_dir / "performance/run_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _create_performance_dashboard(self):
        """Create a comprehensive performance dashboard."""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        # Plot 1: Training Loss Evolution
        if len(self.epoch_df) > 0:
            train_loss = self.epoch_df[self.epoch_df['metric'] == 'train_loss']
            if len(train_loss) > 0:
                self._plot_metric_evolution(train_loss, axes[0], 'Training Loss', 'blue')
        
        # Plot 2: Final Test Metrics
        if len(self.results_df) > 0:
            self._plot_final_metrics_summary(axes[1])
        
        # Plot 3: Run Consistency
        if len(self.epoch_df) > 0:
            self._plot_run_consistency(axes[2])
        
        # Plot 4: Gradient Health Summary
        if len(self.batch_df) > 0:
            self._plot_gradient_health_summary(axes[3])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance/performance_dashboard.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_metric_evolution(self, data, ax, title, color):
        """Plot metric evolution on given axis."""
        
        runs = data['trial_run_id'].unique()
        colors = plt.cm.tab10(range(len(runs)))
        
        for i, run_id in enumerate(runs):
            run_data = data[data['trial_run_id'] == run_id]
            ax.plot(run_data['epoch'], run_data['value'],
                   alpha=0.7, linewidth=2, color=colors[i],
                   label=f'Run {i+1}')
        
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_final_metrics_summary(self, ax):
        """Plot final metrics summary on given axis."""
        
        if len(self.results_df) == 0:
            ax.text(0.5, 0.5, 'No final metrics available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Final Metrics')
            return
        
        metrics = self.results_df['metric'].tolist()
        values = self.results_df['value'].tolist()
        
        bars = ax.bar(metrics, values, alpha=0.7)
        ax.set_title('Final Test Metrics')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
    
    def _plot_run_consistency(self, ax):
        """Plot run consistency analysis on given axis."""
        
        if len(self.epoch_df) == 0:
            ax.text(0.5, 0.5, 'No epoch data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Run Consistency')
            return
        
        # Focus on training loss for consistency analysis
        train_loss = self.epoch_df[self.epoch_df['metric'] == 'train_loss']
        
        if len(train_loss) == 0:
            ax.text(0.5, 0.5, 'No training loss data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Run Consistency')
            return
        
        # Calculate final loss for each run
        runs = train_loss['trial_run_id'].unique()
        final_losses = []
        
        for run_id in runs:
            run_data = train_loss[train_loss['trial_run_id'] == run_id]
            final_epoch = run_data['epoch'].max()
            final_loss = run_data[run_data['epoch'] == final_epoch]['value'].iloc[0]
            final_losses.append(final_loss)
        
        # Create box plot - fix the dimension issue
        if len(final_losses) > 0:
            # For boxplot, we need to pass data as a list of lists
            ax.boxplot([final_losses], labels=['All Runs'])
            ax.set_title('Run Consistency (Final Training Loss)')
            ax.set_ylabel('Final Training Loss')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No final loss data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Run Consistency')
    
    def _plot_gradient_health_summary(self, ax):
        """Plot gradient health summary on given axis."""
        
        if len(self.batch_df) == 0:
            ax.text(0.5, 0.5, 'No gradient data available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Gradient Health')
            return
        
        # Get gradient metrics
        gradient_metrics = [m for m in self.batch_df['metric'].unique() if 'gradient' in m]
        
        if len(gradient_metrics) == 0:
            ax.text(0.5, 0.5, 'No gradient metrics available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Gradient Health')
            return
        
        # Calculate mean values for each gradient metric
        metric_means = []
        metric_names = []
        
        for metric in gradient_metrics:
            metric_data = self.batch_df[self.batch_df['metric'] == metric]
            mean_val = metric_data['value'].mean()
            metric_means.append(mean_val)
            metric_names.append(metric.replace('gradient_', '').replace('_', ' ').title())
        
        # Create bar plot
        bars = ax.bar(metric_names, metric_means, alpha=0.7, color='orange')
        ax.set_title('Gradient Health Summary')
        ax.set_ylabel('Mean Value')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, metric_means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)

def calculate_epoch_average_with_sample_size(df):
    """Calculate epoch averages across trials, where each trial is averaged over its runs"""
    if len(df) == 0:
        return pd.DataFrame(columns=['epoch', 'mean', 'std', 'n_trials'])
    
    # First, average each trial's runs for each epoch
    trial_epoch_averages = []
    
    for trial_name in df['trial_name'].unique():
        trial_data = df[df['trial_name'] == trial_name]
        
        # Group by epoch and average across runs for this trial
        trial_epoch_agg = trial_data.groupby('epoch')['value'].agg(['mean', 'std', 'count']).reset_index()
        trial_epoch_agg['trial_name'] = trial_name
        trial_epoch_averages.append(trial_epoch_agg)
    
    if not trial_epoch_averages:
        return pd.DataFrame(columns=['epoch', 'mean', 'std', 'n_trials'])
    
    # Combine all trial averages
    all_trial_averages = pd.concat(trial_epoch_averages, ignore_index=True)
    
    # Now calculate cross-trial averages for each epoch
    all_epochs = sorted(all_trial_averages['epoch'].unique())
    results = []
    
    for epoch in all_epochs:
        # Get all trials that have data for this epoch
        epoch_data = all_trial_averages[all_trial_averages['epoch'] == epoch]
        
        if len(epoch_data) > 0:
            # Calculate statistics across trials (each trial is already averaged over its runs)
            mean_val = epoch_data['mean'].mean()  # Average of trial averages
            std_val = epoch_data['mean'].std() if len(epoch_data) > 1 else 0  # Std across trials
            n_trials = len(epoch_data)
            
            results.append({
                'epoch': epoch,
                'mean': mean_val,
                'std': std_val,
                'n_trials': n_trials
            })
    
    return pd.DataFrame(results)

def get_dynamic_epoch_range(df):
    """Find the actual epoch range across all trials"""
    if len(df) == 0:
        return {'global_min_epoch': 0, 'global_max_epoch': 0, 'trial_ranges': {}, 'unique_epochs': []}
    
    epoch_stats = df.groupby('trial_name')['epoch'].agg(['min', 'max', 'count']).reset_index()
    
    return {
        'global_min_epoch': epoch_stats['min'].min(),
        'global_max_epoch': epoch_stats['max'].max(),
        'trial_ranges': epoch_stats.set_index('trial_name').to_dict('index'),
        'unique_epochs': sorted(df['epoch'].unique())
    }

def find_common_epochs(df):
    """Find epochs where all trials have data"""
    if len(df) == 0:
        return []
    
    trial_epochs = {}
    for trial_name in df['trial_name'].unique():
        trial_data = df[df['trial_name'] == trial_name]
        trial_epochs[trial_name] = set(trial_data['epoch'].unique())
    
    if not trial_epochs:
        return []
    
    # Find intersection of all trial epochs
    common_epochs = set.intersection(*trial_epochs.values())
    return sorted(list(common_epochs))

def get_trial_statistics(df):
    """Get statistics about trial completion"""
    if len(df) == 0:
        return pd.DataFrame()
    
    stats = []
    for trial_name in df['trial_name'].unique():
        trial_data = df[df['trial_name'] == trial_name]
        epoch_range = trial_data['epoch']
        
        stats.append({
            'trial_name': trial_name,
            'min_epoch': epoch_range.min(),
            'max_epoch': epoch_range.max(),
            'total_epochs': len(epoch_range.unique()),
            'completion_percent': (epoch_range.max() / 100) * 100 if epoch_range.max() <= 100 else 100
        })
    
    return pd.DataFrame(stats)

def normalize_to_completion_percentage(df):
    """Normalize epochs to completion percentage (0-100%)"""
    if len(df) == 0:
        return df
    
    df_normalized = df.copy()
    
    for trial_name in df['trial_name'].unique():
        trial_data = df[df['trial_name'] == trial_name]
        max_epoch = trial_data['epoch'].max()
        
        # Convert epochs to percentage of completion
        mask = df_normalized['trial_name'] == trial_name
        df_normalized.loc[mask, 'epoch_percent'] = (
            df_normalized.loc[mask, 'epoch'] / max_epoch * 100
        )
    
    return df_normalized

def handle_missing_epochs(df, strategy='forward_fill'):
    """Handle missing epochs with different strategies"""
    if len(df) == 0:
        return df
    
    if strategy == 'forward_fill':
        # Use last known value for missing epochs
        result_dfs = []
        for trial_name in df['trial_name'].unique():
            trial_data = df[df['trial_name'] == trial_name].copy()
            trial_data = trial_data.set_index('epoch')
            
            # Reindex to fill missing epochs
            min_epoch = trial_data.index.min()
            max_epoch = trial_data.index.max()
            full_range = range(min_epoch, max_epoch + 1)
            
            trial_data = trial_data.reindex(full_range)
            trial_data = trial_data.fillna(method='ffill')
            trial_data = trial_data.reset_index()
            trial_data['trial_name'] = trial_name
            trial_data['is_interpolated'] = trial_data['value'].isna()
            
            result_dfs.append(trial_data)
        
        return pd.concat(result_dfs, ignore_index=True)
    
    elif strategy == 'exclude_missing':
        # Only use epochs where all trials have data
        common_epochs = find_common_epochs(df)
        return df[df['epoch'].isin(common_epochs)]
    
    else:
        return df

def create_cross_trial_comparison(batch_df, epoch_df, results_df, base_dir):
    """Create cross-trial comparison plots and analysis with dynamic epoch handling."""
    
    print("\n4. Creating cross-trial comparison...")
    
    # Create comparison directory
    comparison_dir = Path(base_dir) / "analysis" / "cross_trial_comparison"
    comparison_dir.mkdir(exist_ok=True)
    
    # Get dynamic epoch information
    epoch_info = get_dynamic_epoch_range(epoch_df)
    trial_stats = get_trial_statistics(epoch_df)
    
    print(f"  Epoch range: {epoch_info['global_min_epoch']} - {epoch_info['global_max_epoch']}")
    print(f"  Trials with different epoch counts detected: {len(trial_stats['total_epochs'].unique()) > 1}")
    
    # Save epoch statistics
    trial_stats.to_csv(comparison_dir / "trial_epoch_statistics.csv", index=False)
    print(f"  [OK] Saved trial epoch statistics")
    
    # Create adaptive visualizations
    create_adaptive_visualizations(epoch_df, comparison_dir, epoch_info, trial_stats)
    
    # Create cross-trial averaged plots with proper sample size handling
    create_cross_trial_averaged_plots(epoch_df, comparison_dir)
    
    # Extract trial information from trial names
    trial_info = []
    for trial_name in batch_df['trial_name'].unique():
        print(f"  Parsing trial: {trial_name}")
        
        # Parse trial name format: test_trial_lr_0.1_batch_4_eps_1e-5
        try:
            parts = trial_name.split('_')
            if len(parts) >= 6 and parts[0] == 'test' and parts[1] == 'trial':
                # Format: test_trial_lr_0.1_batch_4_eps_1e-5
                lr_str = parts[3]  # '0.1'
                batch_str = parts[5]  # '4'
                eps_str = parts[7]  # '1e-5'
                
                lr = float(lr_str)
                batch_size = int(batch_str)
                seq_len = 50  # Default sequence length for test trials
                eps = float(eps_str)
            elif len(parts) >= 8:
                # Format: trial_lr_0.01_batch_4_len_50_eps_1e-3
                lr_str = parts[2]  # '0.01'
                batch_str = parts[4]  # '4'
                len_str = parts[6]  # '50'
                eps_str = parts[8]  # '1e-3'
                
                lr = float(lr_str)
                batch_size = int(batch_str)
                seq_len = int(len_str)
                eps = float(eps_str)
            elif len(parts) >= 6:
                # Fallback for old format without eps
                lr_str = parts[2]  # '0.01'
                batch_str = parts[4]  # '4'
                len_str = parts[6]  # '50'
                
                lr = float(lr_str)
                batch_size = int(batch_str)
                seq_len = int(len_str)
                eps = 0.0  # Default epsilon
            else:
                print(f"    Warning: Could not parse trial name format: {trial_name}")
                continue
        except (ValueError, IndexError) as e:
            print(f"    Warning: Error parsing trial name '{trial_name}': {e}")
            continue
        
        # Get final performance metrics
        trial_results = results_df[results_df['trial_name'] == trial_name] if len(results_df) > 0 else pd.DataFrame()
        final_test_acc = trial_results[trial_results['metric'] == 'test_acc']['value'].mean() if len(trial_results) > 0 else 0
        final_test_loss = trial_results[trial_results['metric'] == 'test_loss']['value'].mean() if len(trial_results) > 0 else float('inf')
        
        # Get gradient health metrics
        trial_batch = batch_df[batch_df['trial_name'] == trial_name]
        gradient_l2_norm = trial_batch[trial_batch['metric'] == 'gradient_l2_norm']['value'].mean() if len(trial_batch) > 0 else 0
        
        # Get epoch statistics for this trial
        trial_epoch_stats = trial_stats[trial_stats['trial_name'] == trial_name]
        max_epoch = trial_epoch_stats['max_epoch'].iloc[0] if len(trial_epoch_stats) > 0 else 0
        total_epochs = trial_epoch_stats['total_epochs'].iloc[0] if len(trial_epoch_stats) > 0 else 0
        
        trial_info.append({
            'trial_name': trial_name,
            'lr': lr,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'eps': eps,
            'test_acc': final_test_acc,
            'test_loss': final_test_loss,
            'gradient_l2_norm': gradient_l2_norm,
            'max_epoch': max_epoch,
            'total_epochs': total_epochs,
            'early_stopped': max_epoch < 100  # Assuming 100 is the expected max
        })
        
        print(f"    Parsed: LR={lr}, Batch={batch_size}, SeqLen={seq_len}, Epochs={total_epochs}")
    
    if len(trial_info) == 0:
        print("  [ERROR] No valid trials found for comparison")
        return
    
    trial_summary = pd.DataFrame(trial_info)
    print(f"  [OK] Created summary for {len(trial_summary)} trials")
    
    # Create performance comparison plots with epoch awareness
    create_performance_comparison_plots_adaptive(trial_summary, comparison_dir, epoch_info)
    
    # Create gradient health comparison
    create_gradient_health_comparison(trial_summary, comparison_dir)
    
    # Generate overall summary report with early stopping analysis
    generate_overall_summary_report_adaptive(trial_summary, comparison_dir, epoch_info)
    
    print(f"  [OK] Cross-trial comparison completed")

def create_adaptive_visualizations(epoch_df, output_dir, epoch_info, trial_stats):
    """Create visualizations that adapt to data characteristics"""
    
    if len(epoch_df) == 0:
        print("  No epoch data available for adaptive visualizations")
        return
    
    print("  Creating adaptive visualizations...")
    
    # Create multiple visualization approaches
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Raw data with actual ranges
    ax1 = plt.subplot(2, 3, 1)
    plot_raw_data_with_ranges(epoch_df, ax1, epoch_info)
    
    # 2. Common epochs only
    ax2 = plt.subplot(2, 3, 2)
    plot_common_epochs(epoch_df, ax2)
    
    # 3. Normalized progress
    ax3 = plt.subplot(2, 3, 3)
    plot_normalized_progress(epoch_df, ax3)
    
    # 4. Final performance comparison
    ax4 = plt.subplot(2, 3, 4)
    plot_final_performance(epoch_df, ax4)
    
    # 5. Early stopping analysis
    ax5 = plt.subplot(2, 3, 5)
    plot_early_stopping_analysis(epoch_df, ax5, trial_stats)
    
    # 6. Completion statistics
    ax6 = plt.subplot(2, 3, 6)
    plot_completion_statistics(trial_stats, ax6)
    
    plt.tight_layout()
    plt.savefig(output_dir / "adaptive_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Created adaptive visualizations")

def plot_raw_data_with_ranges(epoch_df, ax, epoch_info):
    """Plot all trials with their actual ranges"""
    if len(epoch_df) == 0:
        return
    
    # Plot each trial with its actual epoch range
    for trial_name in epoch_df['trial_name'].unique():
        trial_data = epoch_df[epoch_df['trial_name'] == trial_name]
        
        # Plot each metric
        for metric in trial_data['metric'].unique():
            metric_data = trial_data[trial_data['metric'] == metric]
            if len(metric_data) > 0:
                ax.plot(metric_data['epoch'], metric_data['value'], 
                       label=f"{trial_name}_{metric}", alpha=0.7, linewidth=1)
    
    ax.set_xlim(epoch_info['global_min_epoch'], epoch_info['global_max_epoch'])
    ax.set_title('All Trials - Dynamic Range')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

def plot_common_epochs(epoch_df, ax):
    """Plot only epochs where all trials have data"""
    if len(epoch_df) == 0:
        return
    
    common_epochs = find_common_epochs(epoch_df)
    if len(common_epochs) == 0:
        ax.text(0.5, 0.5, 'No common epochs found', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Common Epochs (None Found)')
        return
    
    # Filter to common epochs
    common_data = epoch_df[epoch_df['epoch'].isin(common_epochs)]
    
    # Plot each trial
    for trial_name in common_data['trial_name'].unique():
        trial_data = common_data[common_data['trial_name'] == trial_name]
        
        for metric in trial_data['metric'].unique():
            metric_data = trial_data[trial_data['metric'] == metric]
            if len(metric_data) > 0:
                ax.plot(metric_data['epoch'], metric_data['value'], 
                       label=f"{trial_name}_{metric}", alpha=0.7)
    
    ax.set_title(f'Common Epochs ({len(common_epochs)} epochs)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Value')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

def plot_normalized_progress(epoch_df, ax):
    """Plot normalized progress (0-100% completion)"""
    if len(epoch_df) == 0:
        return
    
    # Normalize epochs to completion percentage
    df_normalized = normalize_to_completion_percentage(epoch_df)
    
    # Plot normalized progress
    for trial_name in df_normalized['trial_name'].unique():
        trial_data = df_normalized[df_normalized['trial_name'] == trial_name]
        
        for metric in trial_data['metric'].unique():
            metric_data = trial_data[trial_data['metric'] == metric]
            if len(metric_data) > 0:
                ax.plot(metric_data['epoch_percent'], metric_data['value'], 
                       label=f"{trial_name}_{metric}", alpha=0.7)
    
    ax.set_title('Normalized Progress (0-100%)')
    ax.set_xlabel('Completion %')
    ax.set_ylabel('Value')
    ax.set_xlim(0, 100)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

def plot_final_performance(epoch_df, ax):
    """Plot final performance comparison"""
    if len(epoch_df) == 0:
        return
    
    # Get final performance for each trial
    final_performance = []
    for trial_name in epoch_df['trial_name'].unique():
        trial_data = epoch_df[epoch_df['trial_name'] == trial_name]
        max_epoch = trial_data['epoch'].max()
        final_data = trial_data[trial_data['epoch'] == max_epoch]
        
        for metric in final_data['metric'].unique():
            metric_final = final_data[final_data['metric'] == metric]
            if len(metric_final) > 0:
                final_performance.append({
                    'trial_name': trial_name,
                    'metric': metric,
                    'final_value': metric_final['value'].mean(),
                    'epoch': max_epoch
                })
    
    if final_performance:
        final_df = pd.DataFrame(final_performance)
        
        # Create bar plot of final performance
        metrics = final_df['metric'].unique()
        x_pos = np.arange(len(metrics))
        width = 0.8 / len(final_df['trial_name'].unique())
        
        for i, trial_name in enumerate(final_df['trial_name'].unique()):
            trial_final = final_df[final_df['trial_name'] == trial_name]
            values = [trial_final[trial_final['metric'] == metric]['final_value'].iloc[0] 
                     if len(trial_final[trial_final['metric'] == metric]) > 0 else 0 
                     for metric in metrics]
            
            ax.bar(x_pos + i * width, values, width, label=trial_name, alpha=0.7)
        
        ax.set_title('Final Performance Comparison')
        ax.set_xlabel('Metric')
        ax.set_ylabel('Final Value')
        ax.set_xticks(x_pos + width * (len(final_df['trial_name'].unique()) - 1) / 2)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()

def plot_early_stopping_analysis(epoch_df, ax, trial_stats):
    """Plot early stopping analysis"""
    if len(epoch_df) == 0 or len(trial_stats) == 0:
        return
    
    # Create histogram of stopping epochs
    ax.hist(trial_stats['max_epoch'], bins=20, alpha=0.7, edgecolor='black')
    ax.set_title('Early Stopping Distribution')
    ax.set_xlabel('Stopping Epoch')
    ax.set_ylabel('Number of Trials')
    ax.axvline(trial_stats['max_epoch'].mean(), color='red', linestyle='--', 
               label=f'Mean: {trial_stats["max_epoch"].mean():.1f}')
    ax.legend()

def plot_completion_statistics(trial_stats, ax):
    """Plot completion statistics"""
    if len(trial_stats) == 0:
        return
    
    # Create pie chart of completion status
    early_stopped = len(trial_stats[trial_stats['max_epoch'] < 100])
    full_trials = len(trial_stats[trial_stats['max_epoch'] >= 100])
    
    sizes = [early_stopped, full_trials]
    labels = ['Early Stopped', 'Full Trials']
    colors = ['lightcoral', 'lightgreen']
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Trial Completion Status')

def create_cross_trial_averaged_plots(epoch_df, output_dir):
    """Create cross-trial plots with proper averaging that excludes trials without data"""
    
    if len(epoch_df) == 0:
        print("  No epoch data available for cross-trial averaging")
        return
    
    print("  Creating cross-trial averaged plots...")
    
    # Get all unique metrics
    metrics = epoch_df['metric'].unique()
    
    # Create figure for each metric
    for metric in metrics:
        metric_data = epoch_df[epoch_df['metric'] == metric]
        
        # Calculate epoch averages with sample size tracking
        epoch_agg = calculate_epoch_average_with_sample_size(metric_data)
        
        if len(epoch_agg) == 0:
            continue
            
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot the average line
        plt.plot(epoch_agg['epoch'], epoch_agg['mean'], linewidth=2, 
                label=f'Cross-Trial Average (n={epoch_agg["n_trials"].iloc[0]})')
        
        # Add confidence bands
        plt.fill_between(epoch_agg['epoch'],
                        epoch_agg['mean'] - epoch_agg['std'],
                        epoch_agg['mean'] + epoch_agg['std'],
                        alpha=0.3, label='±1 Std Dev')
        
        # Add sample size annotations where it changes
        prev_n = None
        for _, row in epoch_agg.iterrows():
            if prev_n is None or row['n_trials'] != prev_n:
                plt.annotate(f"n={int(row['n_trials'])}", 
                           xy=(row['epoch'], row['mean']), 
                           xytext=(0, 10), textcoords='offset points',
                           ha='center', fontsize=10, 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                prev_n = row['n_trials']
        
        # Plot individual trials (averaged over their runs) for reference
        for trial_name in metric_data['trial_name'].unique():
            trial_data = metric_data[metric_data['trial_name'] == trial_name]
            
            # Average across runs for this trial
            trial_epoch_agg = trial_data.groupby('epoch')['value'].agg(['mean', 'std']).reset_index()
            
            plt.plot(trial_epoch_agg['epoch'], trial_epoch_agg['mean'], 
                    alpha=0.7, linewidth=2, label=f'{trial_name}')
            
            # Add confidence bands for individual trials
            plt.fill_between(trial_epoch_agg['epoch'],
                           trial_epoch_agg['mean'] - trial_epoch_agg['std'],
                           trial_epoch_agg['mean'] + trial_epoch_agg['std'],
                           alpha=0.2)
        
        plt.title(f'{metric} - Cross-Trial Analysis')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.tight_layout()
        plt.savefig(output_dir / f"cross_trial_{metric}.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    [OK] Created cross-trial plot for {metric}")

def create_performance_comparison_plots_adaptive(trial_summary, output_dir, epoch_info):
    """Create performance comparison plots with epoch awareness"""
    
    # Create enhanced performance plots that account for different epoch counts
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Performance vs Epochs (showing early stopping)
    ax1 = axes[0, 0]
    for _, trial in trial_summary.iterrows():
        color = 'red' if trial['early_stopped'] else 'blue'
        ax1.scatter(trial['total_epochs'], trial['test_acc'], 
                   color=color, alpha=0.7, s=100, label=trial['trial_name'])
    
    ax1.set_xlabel('Total Epochs')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Performance vs Training Duration')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Early stopping analysis
    ax2 = axes[0, 1]
    early_stopped = trial_summary[trial_summary['early_stopped']]
    full_trials = trial_summary[~trial_summary['early_stopped']]
    
    if len(early_stopped) > 0:
        ax2.scatter(early_stopped['lr'], early_stopped['test_acc'], 
                   color='red', alpha=0.7, s=100, label='Early Stopped')
    if len(full_trials) > 0:
        ax2.scatter(full_trials['lr'], full_trials['test_acc'], 
                   color='blue', alpha=0.7, s=100, label='Full Trials')
    
    ax2.set_xlabel('Learning Rate')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Early Stopping vs Performance')
    ax2.legend()
    
    # Plot 3: Epoch distribution
    ax3 = axes[1, 0]
    ax3.hist(trial_summary['total_epochs'], bins=20, alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Total Epochs')
    ax3.set_ylabel('Number of Trials')
    ax3.set_title('Epoch Distribution')
    ax3.axvline(trial_summary['total_epochs'].mean(), color='red', linestyle='--',
               label=f'Mean: {trial_summary["total_epochs"].mean():.1f}')
    ax3.legend()
    
    # Plot 4: Performance efficiency (accuracy per epoch)
    ax4 = axes[1, 1]
    trial_summary['efficiency'] = trial_summary['test_acc'] / trial_summary['total_epochs']
    ax4.scatter(trial_summary['total_epochs'], trial_summary['efficiency'], 
               alpha=0.7, s=100)
    ax4.set_xlabel('Total Epochs')
    ax4.set_ylabel('Accuracy per Epoch')
    ax4.set_title('Training Efficiency')
    
    plt.tight_layout()
    plt.savefig(output_dir / "performance_comparison_adaptive.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Created adaptive performance comparison plots")

def generate_overall_summary_report_adaptive(trial_summary, output_dir, epoch_info):
    """Generate overall summary report with early stopping analysis"""
    
    report_path = output_dir / "adaptive_summary_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("ADAPTIVE CROSS-TRIAL ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Epoch statistics
        f.write("EPOCH STATISTICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Global epoch range: {epoch_info['global_min_epoch']} - {epoch_info['global_max_epoch']}\n")
        f.write(f"Total unique epochs: {len(epoch_info['unique_epochs'])}\n")
        f.write(f"Trials with different epoch counts: {len(trial_summary['total_epochs'].unique())}\n\n")
        
        # Early stopping analysis
        f.write("EARLY STOPPING ANALYSIS:\n")
        f.write("-" * 25 + "\n")
        early_stopped = trial_summary[trial_summary['early_stopped']]
        full_trials = trial_summary[~trial_summary['early_stopped']]
        
        f.write(f"Early stopped trials: {len(early_stopped)} ({len(early_stopped)/len(trial_summary)*100:.1f}%)\n")
        f.write(f"Full trials: {len(full_trials)} ({len(full_trials)/len(trial_summary)*100:.1f}%)\n\n")
        
        if len(early_stopped) > 0:
            f.write("Early Stopped Trials:\n")
            for _, trial in early_stopped.iterrows():
                f.write(f"  - {trial['trial_name']}: {trial['total_epochs']} epochs, "
                       f"acc={trial['test_acc']:.3f}\n")
            f.write(f"  Average stopping epoch: {early_stopped['total_epochs'].mean():.1f}\n")
            f.write(f"  Average final accuracy: {early_stopped['test_acc'].mean():.3f}\n\n")
        
        # Performance comparison
        f.write("PERFORMANCE COMPARISON:\n")
        f.write("-" * 25 + "\n")
        if len(early_stopped) > 0 and len(full_trials) > 0:
            f.write(f"Early stopped avg accuracy: {early_stopped['test_acc'].mean():.3f}\n")
            f.write(f"Full trials avg accuracy: {full_trials['test_acc'].mean():.3f}\n")
            f.write(f"Performance difference: {abs(early_stopped['test_acc'].mean() - full_trials['test_acc'].mean()):.3f}\n\n")
        
        # Training efficiency
        f.write("TRAINING EFFICIENCY:\n")
        f.write("-" * 20 + "\n")
        trial_summary['efficiency'] = trial_summary['test_acc'] / trial_summary['total_epochs']
        f.write(f"Most efficient trial: {trial_summary.loc[trial_summary['efficiency'].idxmax(), 'trial_name']}\n")
        f.write(f"Efficiency score: {trial_summary['efficiency'].max():.4f}\n")
        f.write(f"Average efficiency: {trial_summary['efficiency'].mean():.4f}\n\n")
        
        # Recommendations
        f.write("RECOMMENDATIONS:\n")
        f.write("-" * 15 + "\n")
        if len(early_stopped) > len(full_trials):
            f.write("- Early stopping is working effectively\n")
            f.write("- Consider reducing patience for faster training\n")
        else:
            f.write("- Early stopping may need adjustment\n")
            f.write("- Consider increasing patience or reducing min_delta\n")
        
        f.write(f"- Focus on trials with high efficiency scores\n")
        f.write(f"- Monitor early stopped trials for convergence quality\n")
    
    print(f"  [OK] Generated adaptive summary report")

def create_performance_comparison_plots(trial_summary, output_dir):
    """Create performance comparison plots."""
    
    # Aggregate data across epsilon values to avoid duplicates
    aggregated_data = trial_summary.groupby(['lr', 'seq_len']).agg({
        'test_acc': 'mean',
        'test_loss': 'mean',
        'gradient_l2_norm': 'mean'
    }).reset_index()
    
    # Test accuracy by learning rate and sequence length
    plt.figure(figsize=(15, 10))
    
    # Create subplot for each learning rate
    lrs = sorted(aggregated_data['lr'].unique())
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, lr in enumerate(lrs):
        if i >= len(axes):
            break
            
        lr_data = aggregated_data[aggregated_data['lr'] == lr]
        seq_lens = sorted(lr_data['seq_len'].unique())
        
        accuracies = [lr_data[lr_data['seq_len'] == sl]['test_acc'].iloc[0] for sl in seq_lens]
        
        axes[i].bar(range(len(seq_lens)), accuracies, alpha=0.7)
        axes[i].set_title(f'Test Accuracy - LR = {lr}')
        axes[i].set_xlabel('Sequence Length')
        axes[i].set_ylabel('Test Accuracy')
        axes[i].set_xticks(range(len(seq_lens)))
        axes[i].set_xticklabels(seq_lens)
        axes[i].grid(True, alpha=0.3)
        
        # Add value labels
        for j, acc in enumerate(accuracies):
            axes[i].text(j, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Learning rate vs performance heatmap
    plt.figure(figsize=(12, 8))
    
    # Create pivot table for heatmap - aggregate across epsilon values
    # First, group by lr and seq_len and take the mean across different eps values
    aggregated_data = trial_summary.groupby(['lr', 'seq_len']).agg({
        'test_acc': 'mean',
        'test_loss': 'mean',
        'gradient_l2_norm': 'mean'
    }).reset_index()
    
    pivot_data = aggregated_data.pivot(index='lr', columns='seq_len', values='test_acc')
    
    plt.imshow(pivot_data, cmap='viridis', aspect='auto')
    plt.colorbar(label='Test Accuracy')
    plt.title('Test Accuracy Heatmap')
    plt.xlabel('Sequence Length')
    plt.ylabel('Learning Rate')
    plt.xticks(range(len(pivot_data.columns)), pivot_data.columns)
    plt.yticks(range(len(pivot_data.index)), pivot_data.index)
    
    # Add text annotations
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            value = pivot_data.iloc[i, j]
            if not pd.isna(value):
                plt.text(j, i, f'{value:.3f}', ha='center', va='center', color='white')
    
    plt.tight_layout()
    plt.savefig(output_dir / "performance_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_gradient_health_comparison(trial_summary, output_dir):
    """Create gradient health comparison plots."""
    
    # Aggregate data across epsilon values to avoid duplicates
    aggregated_data = trial_summary.groupby(['lr', 'seq_len']).agg({
        'test_acc': 'mean',
        'test_loss': 'mean',
        'gradient_l2_norm': 'mean'
    }).reset_index()
    
    # Gradient L2 norm comparison
    plt.figure(figsize=(15, 10))
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    lrs = sorted(aggregated_data['lr'].unique())
    
    for i, lr in enumerate(lrs):
        if i >= len(axes):
            break
            
        lr_data = aggregated_data[aggregated_data['lr'] == lr]
        seq_lens = sorted(lr_data['seq_len'].unique())
        
        gradient_norms = [lr_data[lr_data['seq_len'] == sl]['gradient_l2_norm'].iloc[0] for sl in seq_lens]
        
        axes[i].bar(range(len(seq_lens)), gradient_norms, alpha=0.7, color='orange')
        axes[i].set_title(f'Gradient L2 Norm - LR = {lr}')
        axes[i].set_xlabel('Sequence Length')
        axes[i].set_ylabel('Gradient L2 Norm')
        axes[i].set_xticks(range(len(seq_lens)))
        axes[i].set_xticklabels(seq_lens)
        axes[i].grid(True, alpha=0.3)
        
        # Add value labels
        for j, norm in enumerate(gradient_norms):
            axes[i].text(j, norm + norm*0.01, f'{norm:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / "gradient_health_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_overall_summary_report(trial_summary, output_dir):
    """Generate overall summary report."""
    
    # Aggregate data across epsilon values to avoid duplicates
    aggregated_data = trial_summary.groupby(['lr', 'seq_len']).agg({
        'test_acc': 'mean',
        'test_loss': 'mean',
        'gradient_l2_norm': 'mean'
    }).reset_index()
    
    report_path = output_dir / "overall_summary_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("LIF Experiment - Overall Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        # Best performing configurations
        f.write("Best Performing Configurations:\n")
        f.write("-" * 30 + "\n")
        
        # Best accuracy
        best_acc_trial = aggregated_data.loc[aggregated_data['test_acc'].idxmax()]
        f.write(f"Best Test Accuracy: {best_acc_trial['test_acc']:.4f}\n")
        f.write(f"  Configuration: LR={best_acc_trial['lr']}, SeqLen={best_acc_trial['seq_len']}\n\n")
        
        # Best loss
        best_loss_trial = aggregated_data.loc[aggregated_data['test_loss'].idxmin()]
        f.write(f"Best Test Loss: {best_loss_trial['test_loss']:.4f}\n")
        f.write(f"  Configuration: LR={best_loss_trial['lr']}, SeqLen={best_loss_trial['seq_len']}\n\n")
        
        # Gradient health analysis
        f.write("Gradient Health Analysis:\n")
        f.write("-" * 25 + "\n")
        
        # Healthiest gradients (closest to 1.0)
        healthiest_gradients = aggregated_data.copy()
        healthiest_gradients['gradient_health'] = abs(healthiest_gradients['gradient_l2_norm'] - 1.0)
        best_gradient_trial = healthiest_gradients.loc[healthiest_gradients['gradient_health'].idxmin()]
        
        f.write(f"Healthiest Gradients: L2 Norm = {best_gradient_trial['gradient_l2_norm']:.4f}\n")
        f.write(f"  Configuration: LR={best_gradient_trial['lr']}, SeqLen={best_gradient_trial['seq_len']}\n\n")
        
        # Learning rate analysis
        f.write("Learning Rate Analysis:\n")
        f.write("-" * 22 + "\n")
        
        for lr in sorted(aggregated_data['lr'].unique()):
            lr_data = aggregated_data[aggregated_data['lr'] == lr]
            avg_acc = lr_data['test_acc'].mean()
            avg_loss = lr_data['test_loss'].mean()
            avg_gradient = lr_data['gradient_l2_norm'].mean()
            
            f.write(f"LR = {lr}:\n")
            f.write(f"  Avg Test Accuracy: {avg_acc:.4f}\n")
            f.write(f"  Avg Test Loss: {avg_loss:.4f}\n")
            f.write(f"  Avg Gradient L2 Norm: {avg_gradient:.4f}\n\n")
        
        # Sequence length analysis
        f.write("Sequence Length Analysis:\n")
        f.write("-" * 25 + "\n")
        
        for seq_len in sorted(aggregated_data['seq_len'].unique()):
            seq_data = aggregated_data[aggregated_data['seq_len'] == seq_len]
            avg_acc = seq_data['test_acc'].mean()
            avg_loss = seq_data['test_loss'].mean()
            avg_gradient = seq_data['gradient_l2_norm'].mean()
            
            f.write(f"SeqLen = {seq_len}:\n")
            f.write(f"  Avg Test Accuracy: {avg_acc:.4f}\n")
            f.write(f"  Avg Test Loss: {avg_loss:.4f}\n")
            f.write(f"  Avg Gradient L2 Norm: {avg_gradient:.4f}\n\n")
        
        # Recommendations
        f.write("Recommendations:\n")
        f.write("-" * 15 + "\n")
        
        # Find balanced configuration (good performance + healthy gradients)
        aggregated_data['balanced_score'] = (
            aggregated_data['test_acc'] * 0.6 + 
            (1.0 / aggregated_data['test_loss']) * 0.3 + 
            (1.0 / abs(aggregated_data['gradient_l2_norm'] - 1.0)) * 0.1
        )
        
        best_balanced = aggregated_data.loc[aggregated_data['balanced_score'].idxmax()]
        
        f.write(f"Recommended Configuration (Balanced):\n")
        f.write(f"  LR={best_balanced['lr']}, SeqLen={best_balanced['seq_len']}\n")
        f.write(f"  Test Accuracy: {best_balanced['test_acc']:.4f}\n")
        f.write(f"  Test Loss: {best_balanced['test_loss']:.4f}\n")
        f.write(f"  Gradient L2 Norm: {best_balanced['gradient_l2_norm']:.4f}\n")

def main():
    """Main analysis function."""
    
    print("=== LIF Comprehensive Analysis - Per-Trial Analysis ===")
    
    # Configuration
    db_path = r'outputs\test_early_stopping\artifacts\experiment.db'
    base_dir = r'outputs\test_early_stopping'
    
    # Step 1: Extract all metrics
    print("\n1. Extracting all metrics...")
    batch_df, epoch_df, results_df = extract_all_metrics(db_path)
    
    if batch_df is None:
        print("[ERROR] Data extraction failed!")
        return
    
    # Step 2: Get unique trials
    trials = batch_df['trial_name'].unique()
    print(f"\n2. Found {len(trials)} trials to analyze")
    
    # Step 3: Create per-trial analysis
    print("\n3. Creating per-trial analysis...")
    
    for i, trial_name in enumerate(trials):
        print(f"\nAnalyzing trial {i+1}/{len(trials)}: {trial_name}")
        
        # Create directory structure
        trial_dir = create_trial_analysis_structure(base_dir, trial_name)
        
        # Create analyzer
        analyzer = PerTrialAnalyzer(trial_name, batch_df, epoch_df, results_df, trial_dir)
        
        # Generate all plots
        analyzer.create_training_progress_plots()
        analyzer.create_gradient_analysis_plots()
        analyzer.create_convergence_plots()
        analyzer.create_performance_summary()
        analyzer.generate_summary_report()
        
        print(f"  [OK] Completed analysis for {trial_name}")
    
    # Step 4: Create cross-trial comparison
    create_cross_trial_comparison(batch_df, epoch_df, results_df, base_dir)
    
    print(f"\n[OK] Comprehensive analysis completed successfully!")
    print(f"Check the 'analysis' directory in {base_dir} for detailed results.")
    print(f"Cross-trial comparison available in: {base_dir}/analysis/cross_trial_comparison/")

if __name__ == "__main__":
    main()