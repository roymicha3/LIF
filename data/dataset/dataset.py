"""
In this file, we define the Dataset class.
"""
import os
import torch
from torch.utils import data
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
    
class OutputType(Enum):
    """
    This class represents the output type
    """
    NORMAL = 0
    TORCH = auto()
    NUMPY = auto()
    

class Dataset(data.Dataset):
    """
    This class is responsible for loading the data .
    """
    def __init__(self, data_type: DataType = DataType.TRAIN, output_type: OutputType = OutputType.TORCH, encoder: Encoder = IdentityEncoder()) -> None:
        self._type          = data_type
        self._output_type   = output_type
        self._encoder       = encoder
        
    
    def get_encoder(self) -> Encoder:
        """
        return the loaders encoder
        """
        return self._encoder
        
    def __len__(self):
        """
        Return the length of the dataset
        """
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        Returns a single item from the dataset
        """
        raise NotImplementedError
    
    @staticmethod
    def load(filename) -> DataSample:
        """
        Load a single existing sample by name
        """
        if not os.path.splitext(filename)[1]:  # If no extension is present
            filename += '.pkl'
        
        return DataSample.deserialize(filename)
    
    @staticmethod
    def get(sample: DataSample, output_type: OutputType = OutputType.TORCH):
        if output_type is OutputType.NORMAL:
            return sample
        
        elif output_type is OutputType.TORCH:
            return sample.to_torch()
        
        elif output_type is OutputType.NUMPY:
            return sample.to_numpy()