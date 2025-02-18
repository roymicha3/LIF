import os
import csv
from typing import Dict, List, Any
from omegaconf import DictConfig

from pipeline.callback.callback import Callback, Metric, MetricCategory
from settings.serializable import YAMLSerializable


@YAMLSerializable.register("MetricsTracker")
class MetricsTracker(Callback, YAMLSerializable):
    """
    Metrics tracker for storing training statistics
    """
    LOG_NAME = "metrics.log"
    
    def __init__(self, work_dir: str):
        super(MetricsTracker, self).__init__()
        super(YAMLSerializable, self).__init__()
        
        self.log_path = os.path.join(work_dir, MetricsTracker.LOG_NAME)
        self.metrics: Dict[str, List[Any]] = {}
        
    def on_epoch_end(self, metrics: Dict[str, Any]) -> bool:
        """Called at the end of each epoch."""
        for key, value in metrics.items():
            metric = Metric(key)
            if metric.category == MetricCategory.TRACKED:
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(value)
        
        return True

    def on_train_end(self, metrics: Dict[str, Any]):
        """Called at the end of training."""
        for key, value in metrics.items():
            metric = Metric(key)
            if metric.category == MetricCategory.TRACKED:
                self.metrics[key] = [value]
        
        # Save the metrics to the log path as a CSV file
        with open(self.log_path, 'w', newline='', encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write the header
            writer.writerow(list(self.metrics.keys()))
            # Write the rows
            max_length = max(len(values) for values in self.metrics.values())
            for i in range(max_length):
                row = [values[i] if i < len(values) else '' for values in self.metrics.values()]
                writer.writerow(row)

    def get_latest(self, key: str, default: Any = None) -> Any:
        return self.metrics.get(key, [default])[-1]
    
    @classmethod
    def from_config(cls, config: DictConfig, env_config: DictConfig):
        """
        Create an instance from a DictConfig.
        """
        return cls(env_config.work_dir)
