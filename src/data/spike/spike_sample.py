"""
This module defines the SpikeSample class which encapsulates
"""
import torch
import matplotlib.pyplot as plt 
from omegaconf import DictConfig
from typing import List, Generator

from data.data_sample import DataSample
from data.spike.spike_data import SpikeData

from experiment_manager import Environment

class SpikeSample(DataSample):
    """
    This class represents a spike sample.
    It encapsulates a single spike item and provides a method to access it.
    """
    def __init__(self, env: Environment, data: List[SpikeData], size, seq_len, label = None) -> None:
        super().__init__(data, label)
        self.env = env
        self.num_of_neurons = size
        self.seq_len = seq_len
    
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

    def to(self, device):
        """
        Move the spike data to the specified device.
        """
        spike_train = self.to_torch()
        return spike_train.to(device)

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
        

    @staticmethod
    def collate_fn(batch):
        """
        Collate function to combine multiple SpikeSample objects into a batch.
        """
        data = [sample[0] for sample in batch]
        labels = [sample[1] for sample in batch]
        
        res = \
            {
                "data": data,
                "labels": torch.tensor(labels),
            }
        
        return res


def build_time_index(batch: list):
    """
    Build a sparse COO tensor for a batch of SpikeSample objects.
    Supports any number of spikes per neuron.
    """
    batch_map = []
    for batch_idx, sample in enumerate(batch):
        time_index_mapping = {}
        for data in sample._data:
            spike_times = data.get_spike_times()
            neuron_index = data.get_index()
            for spike_time in spike_times:
                time_index_mapping.setdefault(spike_time, []).append(neuron_index)
        
        batch_map.append(time_index_mapping)
    
    return batch_map


def digest_batch(batch: List[SpikeSample]) -> Generator[torch.Tensor, None, None]:
    """
    Generator that yields spikes at each time step for the entire batch.
    """
    batch_size = len(batch)
    num_neurons = batch[0].num_of_neurons
    batch_map = build_time_index(batch)
    # batch_spikes shape: [batch_size, seq_len, num_neurons]
    
    for t in range(batch[0].seq_len):
        batch_spikes = torch.zeros((batch_size, num_neurons), dtype=torch.float32)
        for batch_idx, time_index_mapping in enumerate(batch_map):
            batch_spikes[batch_idx, time_index_mapping.get(t, [])] = 1.0
        
        yield batch_spikes
