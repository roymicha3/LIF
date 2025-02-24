from abc import ABC, abstractmethod
from omegaconf import DictConfig
from typing import List

from data.dataset.dataset import Dataset, DataType, OutputType
from data.dataset.dataset_factory import DatasetFactory
from encoders.encoder_factory import EncoderFactory

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
        
    def on_epoch_end(self, epoch_idx, metrics):
        retval = True
        for callback in self.callbacks:
            retval &= callback.on_epoch_end(epoch_idx, metrics)
            
        return retval
    
    def on_end(self, metrics):
        for callback in self.callbacks:
            callback.on_train_end(metrics)
            
    def load_dataset(self, 
                     dataset_config: DictConfig, 
                     env_config: DictConfig, 
                     type_: DataType = DataType.TRAIN) -> Dataset:
        
        encoder_config = dataset_config.encoder
        encoder = EncoderFactory.create(encoder_config.type, encoder_config, env_config)
        dataset = DatasetFactory.create(
            dataset_config.type, dataset_config, 
            type_, 
            OutputType.TORCH, 
            encoder)
        
        return dataset
    
    @abstractmethod
    def run(self, config: DictConfig, env_config: DictConfig):
        pass
    
    @abstractmethod
    def evaluate(self, network, criterion, dataset):
        pass