"""
In this file, we define the DataLoader class.
"""
from typing import Any, List
from enum import Enum, auto
from data.data_sample import DataSample
from encoders.encoder import Encoder
from encoders.identity_encoder import IdentityEncoder

class DataType(Enum):
    """
    This class represents the data type.
    """
    TRAIN = 0
    TEST = auto()
    VALIDATION = auto()

class DataLoader:
    """
    This class is responsible for loading the data from the dataset.
    """
    def __init__(self, batch_size, encoder: Encoder = IdentityEncoder()) -> None:
        self.__batch_size     =    batch_size
        self.__encoder        =    encoder
        
    def get_batch_size(self) -> int:
        """
        return the batch size
        """
        return self.__batch_size
    
    def get_encoder(self) -> Encoder:
        """
        return the loaders encoder
        """
        return self.__encoder
        
    def load(self, type: DataType = DataType.TRAIN) -> List[DataSample]:
        """
        Load the data.
        """
        raise NotImplementedError
        
    def load_batch(self, type: DataType = DataType.TRAIN) -> List[DataSample]:
        """
        Load a batch of data.
        """
        raise NotImplementedError
    
    def __call__(self, type: DataType = DataType.TEST) -> Any:
        """
        get a generator for the data.
        """
        raise NotImplementedError
