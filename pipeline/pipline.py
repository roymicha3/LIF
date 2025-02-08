from abc import ABC, abstractmethod
from omegaconf import DictConfig
from typing import List

from pipeline.callback.callback import Callback

class Pipeline(ABC):
    """
    Abstract base class for a Neural Network Training Pipeline.
    Defines essential methods for any neural network training pipeline.
    """
    
    def __init__(self):
        super().__init__()
        self.callbacks : List[Callback] = []
        
    def register_callback(self, callback: Callback):
        self.callbacks.append(callback)
        
    def on_epoch_end(self, metrics):
        retval = True
        for callback in self.callbacks:
            retval &= callback.on_epoch_end(metrics)
            
        return retval
    
    def on_end(self, metrics):
        for callback in self.callbacks:
            callback.on_train_end(metrics)
            
    
    @abstractmethod
    def load_dataset(self):
        pass
    
    @abstractmethod
    def run(self, config: DictConfig, env_config: DictConfig):
        pass
    
    @abstractmethod
    def evaluate(self, network, criterion, dataset):
        pass