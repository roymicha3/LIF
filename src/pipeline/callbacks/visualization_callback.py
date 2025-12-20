"""
VisualizationCallback - Generates training visualizations.

Uses on_batch_end to receive visualization data directly from batch_metrics,
avoiding the need for intermediate state management in the pipeline.
"""
from typing import Dict, Any, Optional
from omegaconf import DictConfig

from experiment_manager.environment import Environment
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.common.common import Metric
from experiment_manager.pipelines.callbacks.callback import Callback

from analysis.training_visualization import TrainingVisualizer


# Keys for visualization data in CUSTOM_UNTRACKED
VIZ_KEY_INPUTS = "viz_inputs"
VIZ_KEY_KERNEL = "viz_kernel_voltage"
VIZ_KEY_OUTPUTS = "viz_neuron_outputs"
VIZ_KEY_EPOCH = "viz_epoch_idx"
VIZ_KEY_BATCH = "viz_batch_idx"


@YAMLSerializable.register("VisualizationCallback")
class VisualizationCallback(Callback, YAMLSerializable):
    """
    Callback that generates training visualizations.
    
    Receives visualization data via on_batch_end from batch_metrics and generates
    plots (voltage profiles, spike rasters) based on configuration.
    
    The visualization data is NOT persisted to DB/MLflow - only this callback
    receives it, making it efficient for large tensor data.
    
    Features:
        - Capture data from any batch (configurable via capture_batch)
        - Two plot modes: "immediate" (after batch) or "deferred" (at epoch end)
        - Configurable plot frequency (every N epochs)
    """
    
    def __init__(self, 
                 env: Environment,
                 plot_frequency: int = 10,
                 plot_voltage: bool = True,
                 plot_spikes: bool = True,
                 max_samples: int = 4,
                 capture_batch: int = 0,
                 plot_mode: str = "deferred"):
        """
        Initialize VisualizationCallback.
        
        Args:
            env: Environment instance with artifact_dir and logger
            plot_frequency: Generate plots every N epochs (0 = disabled)
            plot_voltage: Generate voltage profile plots
            plot_spikes: Generate spike raster plots
            max_samples: Maximum samples per plot
            capture_batch: Which batch to capture visualization data from (default: 0)
            plot_mode: "immediate" (plot after batch) or "deferred" (plot at epoch end)
        """
        super(VisualizationCallback, self).__init__()
        super(YAMLSerializable, self).__init__()
        
        self.env = env
        self.plot_frequency = plot_frequency
        self.plot_voltage = plot_voltage
        self.plot_spikes = plot_spikes
        self.max_samples = max_samples
        self.capture_batch = capture_batch
        self.plot_mode = plot_mode
        
        # Statistics
        self.plots_generated = 0
        
        # Buffer for deferred plotting (keyed by epoch_idx)
        self._buffered_viz_data: Dict[int, Dict[str, Any]] = {}
        
        # Initialize visualizer
        self.visualizer = TrainingVisualizer(
            artifact_dir=env.artifact_dir,
            logger=env.logger
        )
        
        env.logger.info(
            f"VisualizationCallback initialized: "
            f"freq={plot_frequency}, voltage={plot_voltage}, spikes={plot_spikes}, "
            f"capture_batch={capture_batch}, mode={plot_mode}"
        )
    
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment):
        """Create from YAML config."""
        return cls(
            env=env,
            plot_frequency=config.get("plot_frequency", 10),
            plot_voltage=config.get("plot_voltage", True),
            plot_spikes=config.get("plot_spikes", True),
            max_samples=config.get("max_samples", 4),
            capture_batch=config.get("capture_batch", 0),
            plot_mode=config.get("plot_mode", "deferred"),
        )
    
    def _should_plot(self, epoch_idx: int) -> bool:
        """Check if we should plot for this epoch."""
        if self.plot_frequency <= 0:
            return False
        return epoch_idx % self.plot_frequency == 0
    
    def _extract_viz_data(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Extract visualization data from CUSTOM_UNTRACKED metrics."""
        viz_data = {}
        
        untracked = metrics.get(Metric.CUSTOM_UNTRACKED)
        if not untracked:
            return viz_data
        
        # Handle list of (name, value) tuples
        if isinstance(untracked, list):
            for item in untracked:
                if isinstance(item, tuple) and len(item) == 2:
                    name, value = item
                    if name.startswith("viz_"):
                        viz_data[name] = value
        
        return viz_data
    
    def _generate_plots(self, epoch_idx: int, viz_data: Dict[str, Any]) -> None:
        """Generate all configured plots."""
        batch_idx = viz_data.get(VIZ_KEY_BATCH, 0)
        
        try:
            # Spike raster plot
            if self.plot_spikes and VIZ_KEY_INPUTS in viz_data:
                self.visualizer.plot_input_spikes_raster(
                    inputs=viz_data[VIZ_KEY_INPUTS],
                    epoch_idx=epoch_idx,
                    batch_idx=batch_idx
                )
            
            # Voltage profile plots
            if self.plot_voltage:
                if VIZ_KEY_KERNEL in viz_data:
                    self.visualizer.plot_voltage_profiles(
                        data=viz_data[VIZ_KEY_KERNEL],
                        plot_type="Kernel",
                        epoch_idx=epoch_idx,
                        batch_idx=batch_idx,
                        max_samples=self.max_samples
                    )
                
                if VIZ_KEY_OUTPUTS in viz_data:
                    self.visualizer.plot_voltage_profiles(
                        data=viz_data[VIZ_KEY_OUTPUTS],
                        plot_type="Neuron",
                        epoch_idx=epoch_idx,
                        batch_idx=batch_idx,
                        max_samples=self.max_samples
                    )
            
            self.plots_generated += 1
            self.env.logger.info(f"Generated visualization plots for epoch {epoch_idx}")
            
        except Exception as e:
            self.env.logger.error(f"Visualization failed at epoch {epoch_idx}: {e}")
    
    # ==================== Callback Interface ====================
    
    def on_start(self) -> None:
        """Called when training starts."""
        self.env.logger.info("VisualizationCallback: Training started")
        self.plots_generated = 0
        self._buffered_viz_data.clear()
    
    def on_batch_end(self, batch_idx: int, metrics: Dict[str, Any]) -> bool:
        """
        Called at the end of each batch.
        
        Receives visualization data directly from batch_metrics.
        Depending on plot_mode, either generates plots immediately or buffers
        for epoch-end plotting.
        
        Args:
            batch_idx: The index of the completed batch.
            metrics: Dictionary of metrics including CUSTOM_UNTRACKED with viz data.
            
        Returns:
            bool: True to continue training.
        """
        # Only capture from the specified batch
        if batch_idx != self.capture_batch:
            return True
        
        # Extract visualization data
        viz_data = self._extract_viz_data(metrics)
        
        if not viz_data:
            return True
        
        epoch_idx = viz_data.get(VIZ_KEY_EPOCH, 0)
        
        if self.plot_mode == "immediate":
            # Generate plots right after the batch
            if self._should_plot(epoch_idx):
                self.env.logger.debug(f"Immediate plotting for epoch {epoch_idx}, batch {batch_idx}")
                self._generate_plots(epoch_idx, viz_data)
        else:
            # Buffer for epoch-end plotting (deferred mode)
            self._buffered_viz_data[epoch_idx] = viz_data
            self.env.logger.debug(f"Buffered viz data for epoch {epoch_idx}")
        
        return True
    
    def on_epoch_end(self, epoch_idx: int, metrics: Dict[str, Any]) -> bool:
        """
        Called at the end of each epoch.
        
        In deferred mode, generates plots from buffered visualization data.
        
        Args:
            epoch_idx: Current epoch index
            metrics: Epoch metrics dict (viz data now comes from on_batch_end)
            
        Returns:
            True to continue training, False to stop
        """
        # In deferred mode, generate plots from buffer
        if self.plot_mode == "deferred" and epoch_idx in self._buffered_viz_data:
            if self._should_plot(epoch_idx):
                self._generate_plots(epoch_idx, self._buffered_viz_data[epoch_idx])
            
            # Clean up buffer
            del self._buffered_viz_data[epoch_idx]
        
        return True
    
    def on_end(self, metrics: Dict[str, Any]) -> None:
        """Called when training ends."""
        # Clear any remaining buffered data
        self._buffered_viz_data.clear()
        
        self.env.logger.info(
            f"VisualizationCallback: Training completed. "
            f"Generated {self.plots_generated} visualization sets."
        )
