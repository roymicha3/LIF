"""
This class encodes the data into spike times - via rate encoding.
"""
import numpy as np
from typing_extensions import override

from encoders.encoder import Encoder
from common import Configuration, SPIKE_NS, MODEL_NS
from data.data_sample import DataSample
from data.spike.spike_data import SpikeData
from data.spike.spike_sample import SpikeSample
from tools.utils import SEQ_LEN


class LatencyEncoder(Encoder):
    """
    This class encodes the data into spike times - via single spike encoding.
    """
    @override
    def __call__(self, data: DataSample):
        return self.encode(data)
    
    def __init__(
        self,
        config: Configuration,
        max_value: int) -> None:
        """
        max_value: the maximum value of the input values
        """
        super().__init__()
        
        self._config = config
        self._T              = self._config[SPIKE_NS.T]
        self._dt             = self._config[SPIKE_NS.dt]
        self._num_of_neurons = self._config[MODEL_NS.NUM_INPUTS]
        self._max_value      = max_value

    def _encode_sample(self, sample: DataSample) -> SpikeSample:
        seq_len = SEQ_LEN(self._T, self._dt)
        data = []

        for neuron_idx, neuron in enumerate(sample.get()):
            
            if isinstance(neuron, list):
                neuron = neuron[0]
            spike_delay = int((neuron / self._max_value) * seq_len)
            
            # if the neuron is silent
            if spike_delay == -1:
                continue
            
            elif spike_delay >= 0:
                spikes = np.array([spike_delay]).astype(int)
            # if no spikes were fired
            else:
                continue

            data.append(SpikeData(self._config, neuron_idx, spikes))

        return SpikeSample(self._config, data, seq_len, sample.get_label())
    
    @override
    def encode(self, data: DataSample) -> SpikeSample:
        """
        encode the data into spike times
        """
        return self._encode_sample(data)
        
    
