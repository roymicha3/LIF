"""
This module defines the SpikeSample class which encapsulates
"""
import torch
import matplotlib.pyplot as plt 
from typing import List

from data.data_sample import DataSample
from data.spike.spike_data import SpikeData

class SpikeSample(DataSample):
    """
    This class represents a spike sample.
    It encapsulates a single spike item and provides a method to access it.
    """
    def __init__(self, data: List[SpikeData], seq_len = None, label = None) -> None:
        super().__init__(data, label)
        self._num_of_neurons = len(data)
        self._seq_len = seq_len
        
    def to_torch(self):
        input_size = self._num_of_neurons
        spike_train = torch.zeros((self._seq_len, input_size), dtype=torch.float32)
        
        for n in range(input_size):
            spike_times = self._data[n].get_spike_times()
            neuron_index = self._data[n].get_index()
            for spike in spike_times:
                spike_train[spike, neuron_index] = 1.0
                
        return spike_train

    def __str__(self):
        return f"SpikeSample({self.get_label()})"
    

    def plot(self) -> None:
        """
        Plot the spike sample.
        """
        plot = plt.figure(figsize=(10, 6))
        
        # Create a subplot for the new neuron's raster plot
        ax = plot.add_subplot(111)
        
        for data in self.get():
            data.plot(ax)
            
        # Customize the plot (you can adjust these settings as needed)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron Index")
        ax.set_yticks([])  # Hide y-axis ticks
            
        plt.show()
