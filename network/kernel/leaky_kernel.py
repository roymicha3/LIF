import torch
import torch.nn as nn
from omegaconf import DictConfig

from network.kernel.kernel import Kernel
from data.spike.spike_sample import SpikeSample
from settings.serializable import YAMLSerializable


@YAMLSerializable.register("LeakyKernel")
class LeakyKernel(Kernel, YAMLSerializable):
    
    def __init__(
        self,
        env_config : DictConfig,
        n,
        tau,
        scale = False,
        learning = False
    ):
        super(LeakyKernel, self).__init__(n, (n, n), learning)
        super(YAMLSerializable, self).__init__()

        self.rnn = nn.RNN(
            n,
            n,
            num_layers=1,
            nonlinearity="relu",
            bias=False,
            batch_first=True,
            dropout=0.0,
            device=env_config.device,
        )
        
        self.env_config = env_config
        self.n = n
        self.dt = env_config.dt
        self.tau = tau
        self.v_0 = env_config.v_0
        self.scale = scale
        self.device = env_config.device
        
        self._beta = 1 - self.dt / self.tau
        
        self._beta_to_weight_hh()
        self._init_weights_hi(1 - self._beta)
        
        # Freeze the RNN parameters by setting them to not require gradients
        self.rnn.requires_grad_(learning)
                    
    def _beta_to_weight_hh(self):
        with torch.no_grad():
            if self._beta is not None:
                # Set all weights to the scalar value of self._beta
                if isinstance(self._beta, float) or isinstance(self._beta, int):
                    self.rnn.weight_hh_l0.copy_(torch.eye(self.rnn.weight_hh_l0.shape[0]) * self._beta)
                elif isinstance(self._beta, torch.Tensor) or isinstance(
                    self._beta, torch.FloatTensor
                ):
                    if len(self._beta) == 1:
                        self.rnn.weight_hh_l0.fill_(self._beta[0])
                elif len(self._beta) == self.hidden_size:
                    # Replace each value with the corresponding value in self._beta
                    for i in range(self.hidden_size):
                        self.rnn.weight_hh_l0.data[i].fill_(self._beta[i])
                else:
                    raise ValueError("Beta must be either a single value or of length 'hidden_size'.")
                    
    def _init_weights_hi(self, x = 1):
        with torch.no_grad():
            # Set weight_ih_l0 to eye
            new_weights = torch.eye(self.rnn.weight_ih_l0.shape[0]) * x
            self.rnn.weight_ih_l0.copy_(new_weights)

    def forward(self, input_):
        input_ = input_.to(self.device)  # Move input to the correct device
        mem = self.rnn(input_)
        
        if self.scale:
            return self.v_0 * mem[0] / self.dt
        
        return mem[0]
    
    @classmethod
    def from_config(cls, config: DictConfig, env_config: DictConfig):
        return cls(
            env_config,
            config.n,
            config.tau,
            scale=config.scale,
            learning=config.learning,
        )
    
    @staticmethod    
    def assimulate_response(input_seq : SpikeSample, tau, dt):
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
                exp = torch.exp(-(times[t:] - t) * dt / tau)
                response[t:, i] += (1 / tau) * exp

        return response
