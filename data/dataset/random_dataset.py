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
from common import Configuration, DATA_NS, MODEL_NS

class RandomDataset(Dataset):
    """
    This class is responsible for loading the data .
    """
    def __init__(self,
                config: Configuration,
                data_type: DataType = DataType.TRAIN,
                output_type: OutputType = OutputType.TORCH,
                encoder: Encoder = ...) -> None:

        super().__init__(data_type, output_type, encoder)
        self._config         = config
        self._input_size     = self._config[MODEL_NS.NUM_INPUTS]
        self._len            = self._config[DATA_NS.DATASET_SIZE]
        self._root           = self._config[DATA_NS.ROOT]
        
        
    def __len__(self):
        """
        Return the length of the dataset
        """
        return self._len
    
    def get_raw(self, idx, encoded = True) -> DataSample:
        """
        Returns a single raw item from the dataset
        """
        filename = f"{idx}.pkl"
        full_path = os.path.join(self._root, filename)
        
        if not os.path.exists(full_path):
            data = np.random.rand(self._input_size).reshape((self._input_size, 1)).tolist()
            label = np.random.randint(0, 2) * 2 - 1 # either +1 or -1 label
            DataSample(data, label).serialize(full_path)
        
        sample = Dataset.load(full_path)
        
        if encoded:
            return self._encoder(sample)
        
        return sample

    def __getitem__(self, idx):
        """
        Returns a single item from the dataset
        """
        sample = self.get_raw(idx)
        return Dataset.get(sample, self._output_type), sample.get_label()
