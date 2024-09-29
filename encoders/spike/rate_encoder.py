"""
This class encodes the data into spike times - via rate encoding.
"""
import numpy as np
from typing_extensions import override

from common import ATTR, SPIKE_NS, MODEL_NS
from encoders.encoder import Encoder
from tools.utils import poisson_events, SEQ_LEN
from data.spike.spike_data import SpikeData
from data.spike.spike_sample import SpikeSample
from data.data_sample import DataSample


class RateEncoder(Encoder):
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
        
        self._T              = ATTR(SPIKE_NS.T)
        self._dt             = ATTR(SPIKE_NS.dt)
        self._num_of_neurons = ATTR(MODEL_NS.NUM_INPUTS)
        self._firing_rate    = firing_rate
        self._random         = random

    def _encode_sample(self, sample: DataSample) -> SpikeSample:
        MS_IN_SECOND = 1000
        max_spikes_in_trial = self._firing_rate * (self._T / MS_IN_SECOND)
        seq_len = SEQ_LEN(self._T, self._dt)
        
        if len(data.shape) == 3:
            assert self._num_of_neurons == data.shape[1] * data.shape[2]
            normalize_factor = max(np.max(sample.get()), 1.0e-6)
            data = data.reshape((len(data), self._num_of_neurons)) / normalize_factor
        
        res = []

        for neuron_idx, neuron in enumerate(data):
            # if the neuron is silent
            if neuron == 0:
                continue
            
            num_of_spikes = int(neuron * max_spikes_in_trial)
            if self._random and num_of_spikes > 0:
                spikes = poisson_events(num_of_spikes, seq_len)
            elif num_of_spikes > 0:
                spikes = np.arange(0, self._T, self._T / num_of_spikes).astype(int)
            # if no spikes were fired
            else:
                continue

            res.append(SpikeData(neuron_idx, spikes))

        return SpikeSample(res, seq_len, sample.get_label())
    
    @override
    def encode(self, sample: DataSample) -> SpikeSample:
        """
        encode the data into spike times
        """
        return self._encode_sample(sample)
    
