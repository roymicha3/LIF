"""
This module defines the SpikeSample class which encapsulates
"""
import matplotlib.pyplot as plt 
from typing import List

from data.data_sample import DataSample
from data.spike.spike_data import SpikeData

class SpikeSample(DataSample):
    """
    This class represents a spike sample.
    It encapsulates a single spike item and provides a method to access it.
    """
    def __init__(self, data: List[SpikeData], label = None) -> None:
        super().__init__(data, label)
        self._num_of_neurons = len(data)

    def __str__(self):
        return f"SpikeSample({self._data})"

    def plot(self) -> None:
        """
        Plot the spike sample.
        """
        plot = plt.figure(figsize=(10, 6))
        
        for data in self._data:
            data.plot(plot)
            
        plt.show()
