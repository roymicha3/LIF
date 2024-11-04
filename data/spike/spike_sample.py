"""
Module defining the SpikeSample class for encapsulating spike data.
"""

import torch
import matplotlib.pyplot as plt 
from typing import List, Optional
from copy import copy

from data.data_sample import DataSample
from data.spike.spike_data import SpikeData
from common import Configuration, MODEL_NS

class SpikeSample(DataSample):
    """Encapsulates spike data for neural network processing."""

    def __init__(self, config: Configuration, data: List[SpikeData], seq_len: Optional[int] = None, label: Optional[int] = None) -> None:
        """
        Initialize a SpikeSample instance.

        Args:
            config: Configuration object containing model parameters.
            data: List of SpikeData objects representing spike sequences.
            seq_len: Length of the sequence. Defaults to None.
            label: Label associated with this sample. Defaults to None.
        """
        super().__init__(data, label)
        self._config = config
        self._num_of_neurons = self._config[MODEL_NS.NUM_INPUTS]
        self._seq_len = seq_len or max(spike.get_max_spike_time() for spike in data)
    
    @property
    def seq_len(self) -> int:
        """Return the sequence length."""
        return self._seq_len
    
    @property
    def size(self) -> int:
        """Return the number of neurons."""
        return self._num_of_neurons
    
    def silence(self, cutoff: int):
        """
        Silence all spikes above the given cutoff.

        Args:
            cutoff: Time threshold above which spikes will be silenced.
        """
        new_sample = self.__copy__()
        for spike_seq in new_sample._data:
            spike_seq.silence(cutoff)
            
        return new_sample
        
    def to_torch(self) -> torch.Tensor:
        """
        Convert the spike data to a PyTorch tensor.

        Returns:
            A tensor representation of the spike train.
        """
        spike_train = torch.zeros((self._seq_len, self._num_of_neurons), dtype=torch.float32)
        for data in self._data:
            spike_train[data.get_spike_times(), data.get_index()] = 1.0
        return spike_train

    def __str__(self) -> str:
        """Return a string representation of the SpikeSample."""
        return f"SpikeSample(label={self.get_label()}, neurons={self.size}, seq_len={self.seq_len})"
    
    def plot(self) -> None:
        """Plot the spike sample as a raster plot."""
        _, ax = plt.subplots(figsize=(10, 6))
        for data in self.get():
            data.plot(ax)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Neuron Index")
        ax.set_yticks([])
        plt.title("Spike Raster Plot")
        plt.show()

    def __copy__(self):
        """
        Create a shallow copy of the SpikeSample instance.

        Returns:
            A new SpikeSample instance with copied attributes.
        """
        return SpikeSample(
            self._config,
            [copy(spike_data) for spike_data in self._data],
            self._seq_len,
            self.get_label()
        )