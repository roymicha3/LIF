"""
In this file, we define the Dataset class.
"""
import os
import torch
import numpy as np

from data.dataset.dataset import Dataset, DataType, OutputType
from data.data_sample import DataSample
from encoders.encoder import Encoder
from encoders.identity_encoder import IdentityEncoder

class RandomDataset(Dataset):
    """
    This class is responsible for loading the data .
    """
    def __init__(self, 
                input_size: int,
                length: int,
                data_type: DataType = DataType.TRAIN,
                output_type: OutputType = OutputType.TORCH,
                encoder: Encoder = ...,
                root = os.path.join("data", "data")) -> None:

        super().__init__(data_type, output_type, encoder)
        self._input_size = input_size
        self._len = length
        self._root = root
        
        
    def __len__(self):
        """
        Return the length of the dataset
        """
        return self._len
    
    def get_raw(self, idx):
        """
        Returns a single raw item from the dataset
        """
        filename = f"{idx}.pkl"
        full_path = os.path.join(self._root, filename)
        
        if not os.path.exists(full_path):
            data = np.random.rand(self._len).reshape((self._len, 1)).tolist()
            label = np.random.randint(0, 1)
            DataSample(data, label).serialize(full_path)
        
        sample = Dataset.load(full_path)
        return self._encoder(sample)

    def __getitem__(self, idx):
        """
        Returns a single item from the dataset
        """
        return Dataset.get(self.get_raw(idx), self._output_type)