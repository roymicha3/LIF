import torch
# from typing import override

from network.kernel.kernel import Kernel
from network.kernel.leaky_kernel import LeakyKernel
from common import SPIKE_NS
from data.spike.spike_sample import SpikeSample

# Define the DEN Node class (assuming the class is provided as is)
class DENKernel(Kernel):
    def __init__(
        self,
        config,
        n,
        device=None,
        dtype=None,
        scale = False,
        learning = False
    ):
        super(DENKernel, self).__init__(n, (n, n), learning)
        
        self.device = device
        self._config = config
        tau_m = self._config[SPIKE_NS.tau_m]
        self._coductness = LeakyKernel(self._config, n, device, dtype, scale=scale, learning=learning, tau=tau_m)
        
        tau_s = self._config[SPIKE_NS.tau_s]
        self._voltage = LeakyKernel(self._config, n, device, dtype, scale=False, learning=learning, tau=tau_m / 4) #TODO: change back
        
    # @override
    def forward(self, x):
        """
        forward function for the layer
        """
        with torch.no_grad():
            x = x.to(self.device)  # Move input to the correct device
            mem = self._coductness(x)
            mem = self._voltage(mem)
        
        return mem
    
    @staticmethod    
    def assimulate_response(input_seq : SpikeSample, tau_m, tau_s, dt):
        """
        this function assimulate how the output of the model should behave
        """
        seq_length = input_seq.seq_len
        input_size = input_seq.size

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
