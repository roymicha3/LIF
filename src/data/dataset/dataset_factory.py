from omegaconf import DictConfig

from encoders.encoder import Encoder
from settings.factory import Factory
from settings.serializable import YAMLSerializable
from data.dataset.dataset import DataType, OutputType

# import all the datesets:
from data.dataset.random_dataset import RandomDataset


class DatasetFactory(Factory):
    """
    Factory class for creating datasets.
    """
    
    @staticmethod
    def create(name, config: DictConfig, 
               data_type: DataType, 
               output_type: OutputType, 
               encoder: Encoder):
        
        return YAMLSerializable.get_by_name(name).from_config(config,
                                                              data_type,
                                                              output_type,
                                                              encoder)