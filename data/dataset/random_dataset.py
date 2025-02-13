"""
In this file, we define the Dataset class.
"""
import os
import numpy as np
from omegaconf import DictConfig

from data.dataset.dataset import Dataset, DataType, OutputType
from data.data_sample import DataSample
from settings.serializable import YAMLSerializable
from encoders.encoder import Encoder

@YAMLSerializable.register("RandomDataset")
class RandomDataset(Dataset):
    """
    This class is responsible for loading the data .
    """
    def __init__(self,
                 input_size: int,
                 len_: int,
                 root: str,
                 data_type: DataType = DataType.TRAIN,
                 output_type: OutputType = OutputType.TORCH,
                 encoder: Encoder = ...) -> None:

        super().__init__(data_type, output_type, encoder)
        self.input_size     = input_size
        self.len            = len_
        self.root           = root
        
        
    def __len__(self):
        """
        Return the length of the dataset
        """
        return self.len
    
    def get_raw(self, idx, encoded = True) -> DataSample:
        """
        Returns a single raw item from the dataset
        """
        filename = f"{idx}.pkl"
        full_path = os.path.join(self.root, filename)
        
        if not os.path.exists(full_path):
            data = np.random.rand(self.input_size).reshape((self.input_size, 1)).tolist()
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
    
    @staticmethod
    def from_config(config: DictConfig, 
                    data_type: DataType, 
                    output_type: OutputType, 
                    encoder: Encoder):
        
        return RandomDataset(config.input_size,
                            config.len,
                            config.root,
                            data_type=data_type,
                            output_type=output_type,
                            encoder=encoder) 
