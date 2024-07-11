"""
This class encodes the data into spike times - via rate encoding.
"""
import numpy as np
from typing import List
from typing_extensions import override

from encoders.encoder import Encoder
from common import ATTR, SPIKE_NS, MODEL_NS
from data.spike.spike_data import SpikeData
from data.spike.spike_sample import SpikeSample
from encoders.spike.spike_utils import NUM_TIME_SAMPLES


class SingleSpikeEncoder(Encoder):
    """
    This class encodes the data into spike times - via single spike encoding.
    """
    @override
    def __call__(self, data: np.array):
        return self.encode(data)
    
    def __init__(
        self,
        max_value: int) -> None:
        """
        max_value: the maximum value of the input values
        """
        super().__init__()
        
        self.__T              = ATTR(SPIKE_NS.T)
        self.__dt             = ATTR(SPIKE_NS.dt)
        self.__num_of_neurons = ATTR(MODEL_NS.NUM_INPUTS)
        self.__max_value      = max_value

    def _encode_sample(self, sample) -> SpikeSample:
        time_samples = NUM_TIME_SAMPLES(self.__T, self.__dt)
        data = []

        for neuron_idx, neuron in enumerate(sample):
            spike_delay = int((neuron / self.__max_value) * time_samples)
            
            # if the neuron is silent
            if spike_delay == 0:
                continue
            elif spike_delay > 0:
                spikes = np.array([spike_delay]).astype(int)
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
        assert len(data.shape) == 2
        assert self.__num_of_neurons == data.shape[1]
        
        res = [self._encode_sample(sample) for sample in data]
        return res
    
