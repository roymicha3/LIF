import os
from typing import Dict, Any
from omegaconf import DictConfig

from experiment.db.database import DB

from pipeline.callback.callback import Callback, Metric
from settings.serializable import YAMLSerializable


@YAMLSerializable.register("CheckpointCallback")
class CheckpointCallback(Callback, YAMLSerializable):
    """
    Metrics tracker for storing training statistics
    """
    CHECKPOINT_NAME = "checkpoint"
    
    def __init__(self, parent_id, interval: int):
        super(CheckpointCallback, self).__init__()
        super(YAMLSerializable, self).__init__()
        
        self.index = 0
        self.current_checkpoint = 0
        
        self.interval = interval
        self.parent_id = parent_id
        
        self.checkpoint_path = os.path.join(DB.instance().data_path, CheckpointCallback.CHECKPOINT_NAME)
        
    def on_epoch_end(self, epoch_idx, metrics: Dict[str, Any]) -> bool:
        """Called at the end of each epoch."""
        
        self.index += 1
        if self.index % self.interval == 0:
            file_path = self.checkpoint_path + f"{self.current_checkpoint}"
            metrics[Metric.NETWORK].save(file_path)
            self.current_checkpoint += 1
            
            # register artifact to the epoch:
            artifact_id = DB.instance().create_artifact("checkpoint", file_path)
            DB.instance().add_artifact_to_epoch(epoch_idx, self.parent_id, artifact_id)
        
        return True

    def on_train_end(self, metrics: Dict[str, Any]):
        """Called at the end of training."""
        pass

    def get_latest(self, key: str, default: Any = None) -> Any:
        pass
    
    @classmethod
    def from_config(cls, config: DictConfig, env_config: DictConfig, parent_id):
        """
        Create an instance from a DictConfig.
        """
        return cls(parent_id, config.interval)
