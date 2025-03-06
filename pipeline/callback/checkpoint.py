import os
import csv
from typing import Dict, List, Any
from omegaconf import DictConfig

from pipeline.callback.callback import Callback, Metric, MetricCategory
from settings.serializable import YAMLSerializable


@YAMLSerializable.register("CheckpointCallback")
class CheckpointCallback(Callback, YAMLSerializable):
    """
    Metrics tracker for storing training statistics
    """
    CHECKPOINT_NAME = "checkpoint"
    
    def __init__(self, work_dir: str, interval: int):
        super(CheckpointCallback, self).__init__()
        super(YAMLSerializable, self).__init__()
        
        self.index = 0
        self.current_checkpoint = 0
        
        self.interval = interval
        self.checkpoint_path = os.path.join(work_dir, CheckpointCallback.CHECKPOINT_NAME)
        
    def on_epoch_end(self, metrics: Dict[str, Any]) -> bool:
        """Called at the end of each epoch."""
        
        self.index += 1
        if self.index % self.interval == 0:
            file_path = self.checkpoint_path + f"{self.current_checkpoint}"
            metrics[Metric.NETWORK].save(file_path)
            self.current_checkpoint += 1
        
        return True

    def on_train_end(self, metrics: Dict[str, Any]):
        """Called at the end of training."""
        pass

    def get_latest(self, key: str, default: Any = None) -> Any:
        pass
    
    @classmethod
    def from_config(cls, config: DictConfig, env_config: DictConfig):
        """
        Create an instance from a DictConfig.
        """
        return cls(env_config.work_dir, config.interval)
