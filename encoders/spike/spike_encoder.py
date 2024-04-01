"""
This class encodes the data into spike times - via rate encoding.
"""
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from typing_extensions import override

from common import ATTR, SPIKE_NS
from encoders.encoder import Encoder
from encoders.spike.spike_utils import poisson_events
from data.spike.spike_data import SpikeData
from data.spike.spike_sample import SpikeSample


class SpikeEncoder(Encoder):
    """
    This class encodes the data into spike times - via rate encoding.
    """
    @override
    def __call__(self, data: np.array):
        return self.encode(data)
    
    def __init__(
        self,
        firing_rate: int = 20,
        random: bool = True) -> None:
        """
        firing_rate: Firing rate of the neurons in Hz.
        random: If True, spikes are generated randomly. If False, spikes are generated uniformly.
        """
        super().__init__()
        
        self.__T              = ATTR(SPIKE_NS.T)
        self.__dt             = ATTR(SPIKE_NS.dt)
        self.__num_of_neurons = ATTR(SPIKE_NS.NUM_NEURONS)
        self.__firing_rate    = firing_rate
        self.__random         = random

    def _encode_sample(self, sample) -> SpikeSample:
        MS_IN_SECOND = 1000
        max_spikes_in_trial = self.__firing_rate * (self.__T / MS_IN_SECOND)
        time_samples = int(self.__T / self.__dt)
        
        data = []

        for neuron_idx, neuron in enumerate(sample):
            # if the neuron is silent
            if neuron == 0:
                continue
            num_of_spikes = int(neuron * max_spikes_in_trial)
            if self.__random and num_of_spikes > 0:
                spikes = poisson_events(num_of_spikes, time_samples)
            elif num_of_spikes > 0:
                spikes = np.arange(0, self.__T, self.__T / num_of_spikes).astype(int)
            # if no spikes were fired
            else:
                continue

            data.append(SpikeData(neuron_idx, spikes))

        return SpikeSample(data)
    
    @override
    def encode(self, data: np.array) -> List[SpikeSample]:
        """
        encode the data into spike times
        """
        assert len(data.shape) == 3
        assert self.__num_of_neurons == data.shape[1] * data.shape[2]
        
        normalize_factor = max(np.max(data), 1.0e-6)

        flatten_data = data.reshape((len(data), self.__num_of_neurons)) / normalize_factor
        
        res = [self._encode_sample(sample) for sample in flatten_data]
        return res
    
    #TODO: fix this
    def plot_spike_encoding(self, encoded_data: List[SpikeSample]):
        """
        Plot spike times for each neuron.
        :param encoded_data: List of SpikeSample containing the spike times.
        """
        # for spike_data in encoded_data:
        #     plt.figure(figsize=(10, 6))
            
        #     plt.eventplot(
        #         spike_data.spike_times,
        #         lineoffsets=spike_data.neuron_indices,
        #         linelengths=0.5)
            
        #     plt.xlabel('Time (ms)')
        #     plt.ylabel('Neuron Indices')
        #     plt.title('Spike Times for Each Neuron')
        #     plt.show()
        pass
