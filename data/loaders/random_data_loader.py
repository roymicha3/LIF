"""
Definition of a random data generator
"""

# TODO: delete this class, its no longer needed
import torch
import matplotlib.pyplot as plt
from typing import List
import numpy as np

from data.loaders.data_loader import DataLoader, DataType
from data.data_sample import DataSample
from common import Configuration, MODEL_NS, DATA_NS

MAX_SAMPLES = 10000

class RandomDataLoader(DataLoader):
    """
    this class is reposible for loading random data in a given range
    """
    def __init__(self, config: Configuration, batch_size, encoder, max_value: int):
        super().__init__(batch_size, encoder)
        self._config = config
        self._max_value        = max_value
        self._num_of_neurons   = self._config[MODEL_NS.NUM_INPUTS]
        self._num_of_classes   = self._config[MODEL_NS.NUM_CLASSES]
    
    
    def _partial_load(self, size: int) -> List[DataSample]:
        possible_labels = np.arange(0, self._num_of_classes, 1)
        data = []
        for _ in range(size):
            current_data = np.random.uniform(
                low=1,
                high=self._max_value,
                size=self._num_of_neurons)
            
            label = np.random.choice(possible_labels)
            data.append(DataSample(current_data, label))
        
        return self.get_encoder().encode(np.array(data))
    
    
    def load(self, type: DataType = DataType.TRAIN) -> List[DataSample]:
        return self._partial_load(MAX_SAMPLES)
    
    def load_batch(self, type: DataType = DataType.TRAIN) -> List[DataSample]:
        return self._partial_load(self.get_batch_size())
    
    def __call__(self, type: DataType = DataType.TEST) -> List[DataSample]:
        return self.load()
    
    def plot_value_distribution(self):
        """
        Plots the distribution of generated values for each neuron.
        """
        # Load a sample batch
        data = self.load_batch()
        data_array = np.array([sample.data for sample in data])

        # Plot distribution for each neuron
        for i in range(self._num_of_neurons):
            plt.hist(data_array[:, i], bins='auto')
            plt.title(f"Distribution of Neuron {i+1}")
            plt.xlabel("Value")
            plt.ylabel("Frequency")
            plt.show()

    def get_summary(self):
        """
        Provides a summary of the generated data.
        """
        # Load a sample to calculate basic statistics
        data = self.load_batch()
        data_array = np.array([sample.data for sample in data])

        summary = {
            "Max Value": self._max_value,
            "Mean": np.mean(data_array),
            "Standard Deviation": np.std(data_array),
            "Minumum Values": np.min(data_array, axis=0),  # Minimum for each neuron
            "Maximum Values": np.max(data_array, axis=0)   # Maximum for each neuron
        }

        return summary