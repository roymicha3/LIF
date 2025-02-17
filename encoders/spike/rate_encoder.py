"""
This class encodes the data into spike times - via rate encoding.
"""
import numpy as np
from omegaconf import DictConfig
from typing_extensions import override

from encoders.encoder import Encoder
from data.data_sample import DataSample
from data.spike.spike_data import SpikeData
from data.spike.spike_sample import SpikeSample
from tools.utils import poisson_events, SEQ_LEN
from settings.serializable import YAMLSerializable


@YAMLSerializable.register("RateEncoder")
class RateEncoder(Encoder, YAMLSerializable):
    """
    This class encodes the data into spike times - via rate encoding.
    """
    @override
    def __call__(self, data: np.array):
        return self.encode(data)
    
    def __init__(
        self,
        env_config: DictConfig,
        num_of_neurons: int,
        firing_rate: int = 20,
        random: bool = True) -> None:
        """
        firing_rate: Firing rate of the neurons in Hz.
        random: If True, spikes are generated randomly. If False, spikes are generated uniformly.
        """
        super().__init__()
        super(YAMLSerializable, self).__init__()
        
        self.env_config     = env_config
        self.T              = env_config.T
        self.dt             = env_config.dt
        self.num_of_neurons = num_of_neurons
        self.firing_rate    = firing_rate
        self.random         = random

    def _encode_sample(self, sample: DataSample) -> SpikeSample:
        MS_IN_SECOND = 1000
        max_spikes_in_trial = self.firing_rate * (self.T / MS_IN_SECOND)
        seq_len = SEQ_LEN(self.T, self.dt)
        
        if len(data.shape) == 3:
            assert self.num_of_neurons == data.shape[1] * data.shape[2]
            normalize_factor = max(np.max(sample.get()), 1.0e-6)
            data = data.reshape((len(data), self.num_of_neurons)) / normalize_factor
        
        res = []

        for neuron_idx, neuron in enumerate(data):
            # if the neuron is silent
            if neuron == 0:
                continue
            
            num_of_spikes = int(neuron * max_spikes_in_trial)
            if self.random and num_of_spikes > 0:
                spikes = poisson_events(num_of_spikes, seq_len)
            elif num_of_spikes > 0:
                spikes = np.arange(0, self.T, self.T / num_of_spikes).astype(int)
            # if no spikes were fired
            else:
                continue

            res.append(SpikeData(self.env_config, neuron_idx, spikes))

        return SpikeSample(self.env_config, res, self.num_of_neurons, seq_len, sample.get_label())
    
    @override
    def encode(self, sample: DataSample) -> SpikeSample:
        """
        encode the data into spike times
        """
        return self._encode_sample(sample)
    
    @staticmethod
    def from_config(cls, config: DictConfig, env_config: DictConfig):
        return cls(env_config, config.num_of_neurons, config.firing_rate, config.random)
    
