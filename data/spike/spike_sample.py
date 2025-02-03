"""
This module defines the SpikeSample class which encapsulates
"""
import torch
from typing import List
import matplotlib.pyplot as plt 
from omegaconf import DictConfig

from data.data_sample import DataSample
from data.spike.spike_data import SpikeData

class SpikeSample(DataSample):
    """
    This class represents a spike sample.
    It encapsulates a single spike item and provides a method to access it.
    """
    def __init__(self, env_config: DictConfig, data: List[SpikeData], size, seq_len, label = None) -> None:
        super().__init__(data, label)
        self.env_config = env_config
        self.num_of_neurons = size
        self.seq_len = seq_len
    
    @property
    def seq_len(self):
        return self.seq_len
    
    @property
    def size(self):
        return self.num_of_neurons
    
    def silence(self, cutoff: int):
        """
        Silence all the spikes above the given cutoff
        """
        for spike_seq in self._data:
            spike_seq.silence(cutoff)
        
    def to_torch(self):
        input_size = self.num_of_neurons
        spike_train = torch.zeros((self.seq_len, input_size), dtype=torch.float32)
        for data in self._data:
            spike_times = data.get_spike_times()
            neuron_index = data.get_index()
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
        
        ax.set_title("Spike Raster plot")
        
        for data in self.get():
            data.plot(ax)
            
        # Customize the plot (you can adjust these settings as needed)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron Index")
            
        plt.show()
