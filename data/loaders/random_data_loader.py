"""
Definition of a random data generator
"""
import matplotlib.pyplot as plt
from typing import List
import numpy as np

from data_loader import DataLoader, DataType
from data.data_sample import DataSample
from common import ATTR, MODEL_NS

MAX_SAMPLES = 10000

class RandomDataLoader(DataLoader):
    """
    this class is reposible for loading random data in a given range
    """
    def __init__(self, batch_size, encoder, max_value: int):
        super().__init__(batch_size, encoder)
        self.__max_value        = max_value
        self.__num_of_neurons   = ATTR(MODEL_NS.NUM_INPUTS)
    
    
    def _partial_load(self, size: int) -> List[DataSample]:
        data = []
        for _ in range(size):
            current_data = np.random.uniform(
                low=1,
                high=self.__max_value,
                size=self.__num_of_neurons)
            
            data.append(current_data)
        
        return data
    
    
    def load(self, type: DataType = DataType.TRAIN) -> List[DataSample]:
        return self._partial_load(MAX_SAMPLES)
    
    def load_batch(self, type: DataType = DataType.TRAIN) -> List[DataSample]:
        return self._partial_load(self.__batch_size)
    
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
        for i in range(self.__num_of_neurons):
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
            "Max Value": self.__max_value,
            "Mean": np.mean(data_array),
            "Standard Deviation": np.std(data_array),
            "Minumum Values": np.min(data_array, axis=0),  # Minimum for each neuron
            "Maximum Values": np.max(data_array, axis=0)   # Maximum for each neuron
        }

        return summary