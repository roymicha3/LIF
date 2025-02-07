from abc import ABC, abstractmethod
from omegaconf import DictConfig

from encoders.encoder import Encoder
from data.dataset.dataset import Dataset

class Pipeline(ABC):
    """
    Abstract base class for a Neural Network Training Pipeline.
    Defines essential methods for any neural network training pipeline.
    """
    
    @abstractmethod
    def load_dataset(self):
        pass
    
    @abstractmethod
    def run(self, config: DictConfig, env_config: DictConfig):
        pass
    
    @abstractmethod
    def evaluate(self, network, criterion, dataset):
        pass