import numpy as np
import matplotlib.pyplot as plt
from typing_extensions import override

from encoders.encoder import Encoder
from data.spike.spike_sample import SpikeSample

class SpikeEncoder(Encoder):
    
    def __init__(
        self,
        T, 
        dt, 
        num_of_neurons,
        firing_rate: int = 20,
        random: bool = True) -> None:
        
        super().__init__()
        
        self.T              = T
        self.dt             = dt
        self.num_of_neurons = num_of_neurons
        self.firing_rate    = firing_rate
        self.random         = random

    def _encode_sample(self, sample):
        max_spikes_in_trial = self.firing_rate * (self.T / 1000)
        time_samples = int(self.T / self.dt)
        spike_times = []
        spike_indices = []

        for neuron_idx, neuron in enumerate(sample):
            # if the neuron is silent
            if neuron == 0:
                continue
            num_of_spikes = int(neuron * max_spikes_in_trial)
            if self.random and num_of_spikes > 0:
                spikes = poisson_events(num_of_spikes, time_samples)
            elif num_of_spikes > 0:
                spikes = np.arange(0, self.T, self.T / num_of_spikes).astype(int)
            # if no spikes were fired
            else:
                continue
            spike_times.append(spikes)
            spike_indices.append(neuron_idx)

        return SpikeSample(neuron_indices=spike_indices, spike_times=spike_times, T=self.T, num_of_neurons=self.num_of_neurons)
    
    @override
    def encode(self, data: np.array):
        assert(len(data.shape) == 3)
        assert(self.num_of_neurons == data.shape[1] * data.shape[2])
        
        normalize_factor = max(np.max(data), 1.0e-6)

        flatten_data = data.reshape((len(data), self.num_of_neurons)) / normalize_factor
        
        with ThreadPoolExecutor() as executor:
            encoded_data = list(executor.map(self._encode_sample, flatten_data))

        return encoded_data
    
    def plot_spike_encoding(self, encoded_data):
        """
        Plot spike times for each neuron.
        :param encoded_data: List of SpikeSample containing the spike times.
        """
        for spike_data in encoded_data:
            plt.figure(figsize=(10, 6))
            plt.eventplot(spike_data.spike_times, lineoffsets=spike_data.neuron_indices, linelengths=0.5)
            plt.xlabel('Time (ms)')
            plt.ylabel('Neuron Indices')
            plt.title('Spike Times for Each Neuron')
            plt.show()
 

