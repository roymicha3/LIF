import torch
from typing import override

from .node import Node
from .leaky_node import LeakyNode
from common import ATTR, SPIKE_NS
from data.spike.spike_sample import SpikeSample

# Define the DEN Node class (assuming the class is provided as is)
class DENNode(Node):
    def __init__(
        self,
        n,
        device=None,
        dtype=None,
        scale = False,
        learning = False
    ):
        super(DENNode, self).__init__(n, (n, n), learning)
        
        tau_m = ATTR(SPIKE_NS.tau_m)
        self._coductness = LeakyNode(n, device, dtype, scale=scale, learning=learning, tau=tau_m)
        
        tau_s = ATTR(SPIKE_NS.tau_s)
        self._voltage = LeakyNode(n, device, dtype, scale=False, learning=False, tau=tau_s)
        
    @override
    def forward(self, input_):
        """
        forward function for the layer
        """
        mem = self._coductness(input_)
        mem = self._voltage(mem)
        
        return mem
    
    @staticmethod    
    def assimulate_response(input_seq : SpikeSample, tau_m, tau_s, dt):
        """
        this function assimulate how the output of the model should behave
        """
        seq_length = input_seq._seq_len 
        input_size = input_seq.shape()[0]

        # Initialize the response tensor
        response = torch.zeros(seq_length, input_size)
        
        times = torch.arange(seq_length, dtype=torch.float32)
        
        for neuron in input_seq.get():
            spikes = neuron.get_spike_times()
            i = neuron.get_index()
            for t in spikes:
                # Compute the exponential decay terms
                exp_rise = torch.exp(-(times[t:] - t) * dt / tau_m)
                exp_decay = torch.exp(-(times[t:] - t) * dt / tau_s)
                response[t:, i] += (1 / (tau_m - tau_s)) * (exp_rise - exp_decay)

        return response
