import torch
from omegaconf import DictConfig

from network.kernel.kernel import Kernel
from network.kernel.leaky_kernel import LeakyKernel
from common import SPIKE_NS
from data.spike.spike_sample import SpikeSample
from settings.serializable import YAMLSerializable

@YAMLSerializable.register("DENKernel")
class DENKernel(Kernel, YAMLSerializable):
    def __init__(
        self,
        env_config : DictConfig,
        n,
        tau_m,
        tau_s,
        learning = False
    ):
        super(DENKernel, self).__init__(n, (n, n), learning)
        super(YAMLSerializable, self).__init__()
        
        self.env_config = env_config
        self.n = n
        self.dt = env_config.dt
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.v_0 = env_config.v_0
        
        self.device = env_config.device

        self._coductness = LeakyKernel(env_config,
                                       self.n,
                                       self.tau_m,
                                       scale = True,
                                       learning=learning) 
        
        self._voltage = LeakyKernel(env_config,
                                    self.n,
                                    self.tau_s,
                                    scale = False,
                                    learning=learning)
    
    
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
    
    @classmethod
    def from_config(cls, config: DictConfig, env_config: DictConfig):
        """
        Create an instance from a DictConfig.
        """
        return cls(
                   env_config,
                   config.n,
                   config.tau_m,
                   config.tau_s,
                   learning=config.learning)
