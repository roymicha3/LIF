import torch
import torch.nn as nn

from .node import Node
from common import ATTR, SPIKE_NS
from data.spike.spike_sample import SpikeSample

# Define the LeakyNode class (assuming the class is provided as is)
class LeakyNode(Node):
    def __init__(
        self,
        n,
        device=None,
        dtype=None,
        scale = False,
        learning = False,
        tau = None
    ):
        super(LeakyNode, self).__init__(n, (n, n), learning)

        self.rnn = nn.RNN(
            n,
            n,
            num_layers=1,
            nonlinearity="relu",
            bias=False,
            batch_first=False,
            dropout=0.0,
            device=device,
            dtype=dtype
        )
        self._n = n

        self._dt = ATTR(SPIKE_NS.dt)
        self._tau = tau if tau else ATTR(SPIKE_NS.tau)
        self._beta = 1 - self._dt / self._tau
        
        self._scale = scale
        
        # Placeholder methods and attributes
        self.threshold = 0.5  # Arbitrary threshold for spike detection
        self.spike_grad = lambda x: torch.where(x > self.threshold, torch.ones_like(x), torch.zeros_like(x))
        
        self._beta_to_weight_hh()
        self._init_weights_hi(1 - self._beta)
                    
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
        mem = self.rnn(input_)
        
        if self._scale:
            return mem[0] / self._dt
        
        return mem[0]
    
    @staticmethod    
    def assimulate_response(input_seq : SpikeSample, tau, dt):
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
                exp = torch.exp(-(times[t:] - t) * dt / tau)
                response[t:, i] += (1 / tau) * exp

        return response