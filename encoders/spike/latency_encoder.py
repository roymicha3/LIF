"""
This class encodes the data into spike times - via rate encoding.
"""
import numpy as np
from omegaconf import DictConfig
from typing_extensions import override

from tools.utils import SEQ_LEN
from encoders.encoder import Encoder
from data.data_sample import DataSample
from data.spike.spike_data import SpikeData
from data.spike.spike_sample import SpikeSample
from settings.serializable import YAMLSerializable


@YAMLSerializable.register("LatencyEncoder")
class LatencyEncoder(Encoder, YAMLSerializable):
    """
    This class encodes the data into spike times - via single spike encoding.
    """
    @override
    def __call__(self, data: DataSample):
        return self.encode(data)
    
    
    def __init__(self,
                 env_config: DictConfig,
                 size: int,
                 max_value: int) -> None:
        """
        max_value: the maximum value of the input values
        """
        super().__init__()
        super(YAMLSerializable, self).__init__()
        
        self.env_config     = env_config
        self.T              = env_config.T
        self.dt             = env_config.dt
        self.num_of_neurons = size
        self.max_value      = max_value

    def _encode_sample(self, sample: DataSample) -> SpikeSample:
        seq_len = SEQ_LEN(self.T, self.dt)
        data = []

        for neuron_idx, neuron in enumerate(sample.get()):
            
            if isinstance(neuron, list):
                neuron = neuron[0]
            spike_delay = int((neuron / self.max_value) * seq_len)
            
            # if the neuron is silent
            if spike_delay == -1:
                continue
            
            elif spike_delay >= 0:
                spikes = np.array([spike_delay]).astype(int)
            # if no spikes were fired
            else:
                continue

            data.append(SpikeData(self.env_config, neuron_idx, spikes))

        return SpikeSample(self.env_config, data, self.num_of_neurons, seq_len, sample.get_label())
    
    @override
    def encode(self, data: DataSample) -> SpikeSample:
        """
        encode the data into spike times
        """
        return self._encode_sample(data)
    
    @classmethod
    def from_config(cls, config: DictConfig, env_config: DictConfig):
        return cls(env_config, config.size, config.max_value)
        
    
