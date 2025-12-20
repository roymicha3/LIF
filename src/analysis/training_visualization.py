"""
Training-time visualization utilities for spiking neural networks.

This module provides visualization functions for training metrics including:
- Voltage profiles (membrane potential over time)
- Input spike raster plots
- Kernel responses
"""
import os
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.ticker import MaxNLocator


class TrainingVisualizer:
    """
    Handles visualization during training.
    
    This class provides methods to plot voltage profiles, spike rasters,
    and other training-related visualizations.
    """
    
    STYLE_CONFIG = {
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'font.family': 'DejaVu Sans'
    }
    
    def __init__(self, artifact_dir: str, logger=None, style_config: dict = None):
        """
        Initialize the training visualizer.
        
        Args:
            artifact_dir: Directory to save plot artifacts
            logger: Optional logger for logging messages
            style_config: Optional custom style configuration
        """
        self.artifact_dir = artifact_dir
        self.logger = logger
        self.style_config = style_config or self.STYLE_CONFIG
        self._configure_style()
    
    def _configure_style(self):
        """Configure matplotlib global style."""
        sns.set_style("whitegrid")
        plt.rcParams.update(self.style_config)
    
    def _log(self, message: str):
        """Log a message if logger is available."""
        if self.logger:
            self.logger.info(message)
    
    @staticmethod
    def digest_batch_to_tensor(batch: List) -> torch.Tensor:
        """
        Convert a batch of SpikeSample objects to a tensor.
        
        Args:
            batch: List of SpikeSample objects
            
        Returns:
            Tensor of shape (batch, neurons, time)
        """
        from data.spike.spike_sample import digest_batch
        
        generator = digest_batch(batch)
        # Stack time steps: (time, batch, neurons)
        inputs_tensor = torch.stack([spike_seq for spike_seq in generator], dim=0)
        # Permute to (batch, neurons, time)
        inputs_tensor = inputs_tensor.permute(1, 2, 0)
        return inputs_tensor
    
    def plot_voltage_profiles(
        self,
        data: Union[torch.Tensor, np.ndarray],
        plot_type: str,
        epoch_idx: int,
        batch_idx: int,
        max_samples: int = 4,
        max_traces: int = 4
    ) -> List[str]:
        """
        Plot voltage profiles with professional styling.
        
        Args:
            data: Voltage data tensor of shape (batch, neurons, time) or similar
            plot_type: Type of plot (e.g., "Kernel", "Neuron")
            epoch_idx: Current epoch index
            batch_idx: Current batch index
            max_samples: Maximum number of samples to plot
            max_traces: Maximum number of traces per sample
            
        Returns:
            List of saved plot file paths
        """
        saved_paths = []
        
        for i in range(min(len(data), max_samples)):
            fig = plt.figure(figsize=(8, 10))
            try:
                for j in range(min(len(data[i]), max_traces)):
                    ax = fig.add_subplot(max_traces, 1, j + 1)
                    
                    output = data[i][j]
                    if isinstance(output, torch.Tensor):
                        output = output.cpu().detach().numpy()
                    
                    # Plot data with professional styling
                    ax.plot(
                        output,
                        linewidth=1.5,
                        alpha=0.8,
                        color=sns.color_palette("tab10")[j]
                    )
                    
                    # Formatting
                    ax.set_title(f"{plot_type} {i} - Trace {j + 1}", pad=12)
                    ax.set_xlabel("Time Step", labelpad=8)
                    ax.set_ylabel("Membrane Potential (mV)", labelpad=8)
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                    
                    # Add grid and remove top/right spines
                    ax.grid(True, linestyle='--', alpha=0.6)
                    sns.despine(ax=ax, trim=True)
                
                plt.tight_layout(pad=2.0)
                plot_path = os.path.join(
                    self.artifact_dir,
                    f"epoch_{epoch_idx}_batch_{batch_idx}_{plot_type.lower()}_{i}_voltage.svg"
                )
                plt.savefig(plot_path, format='svg')
                saved_paths.append(plot_path)
                self._log(f"Saved {plot_type} voltage plot: {plot_path}")
                
            finally:
                plt.close(fig)
        
        return saved_paths
    
    def plot_input_spikes_raster(
        self,
        inputs: Union[List, torch.Tensor, np.ndarray],
        epoch_idx: int,
        batch_idx: int,
        sample_idx: int = 0
    ) -> str:
        """
        Plot raster plot of input spikes.
        
        Args:
            inputs: Input spike data (List of SpikeSample or tensor)
            epoch_idx: Current epoch index
            batch_idx: Current batch index
            sample_idx: Which sample in the batch to plot
            
        Returns:
            Path to saved plot file
        """
        # Convert inputs to numpy array for plotting
        if isinstance(inputs, list):
            # Assume it's a list of SpikeSample objects
            inputs_tensor = self.digest_batch_to_tensor(inputs)
            inputs_np = inputs_tensor.cpu().detach().numpy()
        elif isinstance(inputs, torch.Tensor):
            inputs_np = inputs.cpu().detach().numpy()
        else:
            inputs_np = inputs
        
        # inputs_np shape: (batch, neurons, time)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get the specified sample
        spike_data = inputs_np[sample_idx]  # Shape: (neurons, time)
        
        # Find spike locations (where value > 0)
        neuron_indices, time_indices = np.where(spike_data > 0)
        
        # Plot spikes
        ax.scatter(
            time_indices,
            neuron_indices,
            s=2,
            c='black',
            marker='|',
            alpha=0.8
        )
        
        ax.set_title(
            f"Input Spikes at Epoch {epoch_idx + 1}, Batch {batch_idx + 1}, Sample {sample_idx + 1}"
        )
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Neuron Index")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Set y-axis limits to show all neurons
        ax.set_ylim(-0.5, spike_data.shape[0] - 0.5)
        
        plt.tight_layout()
        plot_path = os.path.join(
            self.artifact_dir,
            f"epoch_{epoch_idx}_batch_{batch_idx}_input_spikes.svg"
        )
        plt.savefig(plot_path, format='svg')
        self._log(f"Saved input spikes plot: {plot_path}")
        plt.close(fig)
        
        return plot_path

