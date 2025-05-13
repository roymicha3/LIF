import torch
from omegaconf import DictConfig

from network.kernel.kernel import Kernel
from network.kernel.leaky_kernel import LeakyKernel
from data.spike.spike_sample import SpikeSample

from experiment_manager.environment import Environment
from experiment_manager.common.serializable import YAMLSerializable

@YAMLSerializable.register("DENKernel")
class DENKernel(Kernel, YAMLSerializable):
    def __init__(
        self,
        env : Environment,
        n,
        tau_m,
        tau_s,
        learning = False
    ):
        super(DENKernel, self).__init__(n, (n, n), learning)
        super(YAMLSerializable, self).__init__()
        
        self.env = env
        self.n = n
        self.dt = env.args.dt
        self.tau_m = tau_m
        self.tau_s = tau_s
        self.v_0 = env.args.v_0
        
        self.device = env.device

        self._coductness = LeakyKernel(env,
                                       self.n,
                                       self.tau_m,
                                       scale = True,
                                       learning=learning) 
        
        self._voltage = LeakyKernel(env,
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
    def from_config(cls, config: DictConfig, env: Environment) -> "DENKernel":
        """
        Create an instance from a DictConfig.
        """
        return cls(
                   env,
                   config.n,
                   config.tau_m,
                   config.tau_s,
                   learning=config.learning)
