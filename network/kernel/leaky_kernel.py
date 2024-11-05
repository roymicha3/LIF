import torch
import torch.nn as nn

from network.kernel.kernel import Kernel
from common import Configuration, SPIKE_NS
from data.spike.spike_sample import SpikeSample

class LeakyKernel(Kernel):
    """
    A leaky integrate-and-fire neuron model implemented as a recurrent neural network.
    """
    
    V0 = 1  # Resting potential
    
    def __init__(
        self,
        config: Configuration,
        n: int,
        device=None,
        dtype=None,
        scale: bool = False,
        learning: bool = False,
        tau: float = None
    ):
        """
        Initialize the LeakyKernel.

        Args:
            config (Configuration): Configuration object containing model parameters.
            n (int): Number of neurons in the layer.
            device: The device to run the model on (e.g., 'cpu' or 'cuda').
            dtype: The data type for the model parameters.
            scale (bool): Whether to scale the output. Default is False.
            learning (bool): Whether the model is in learning mode. Default is False.
            tau (float): Time constant for the leaky integrator. If None, uses the value from config.
        """
        super(LeakyKernel, self).__init__(n, (n, n), learning)

        self._config = config
        self.rnn = nn.RNN(
            n, n, num_layers=1, nonlinearity="relu", bias=False,
            batch_first=True, dropout=0.0, device=device, dtype=dtype
        )
        self._n = n
        self.device = device
        self._dt = self._config[SPIKE_NS.dt]
        self._tau = tau if tau else self._config[SPIKE_NS.tau]
        self._beta = 1 - self._dt / self._tau
        
        self._scale = scale
        
        self._beta_to_weight_hh()
        self._init_weights_hi(1 - self._beta)
                    
    def _beta_to_weight_hh(self):
        """Set the hidden-to-hidden weights based on the beta value."""
        with torch.no_grad():
            if self._beta is not None:
                if isinstance(self._beta, (float, int)):
                    self.rnn.weight_hh_l0.copy_(torch.eye(self.rnn.hidden_size) * self._beta)
                elif isinstance(self._beta, torch.Tensor):
                    if self._beta.numel() == 1:
                        self.rnn.weight_hh_l0.fill_(self._beta.item())
                    elif self._beta.numel() == self.rnn.hidden_size:
                        self.rnn.weight_hh_l0.copy_(torch.diag(self._beta))
                    else:
                        raise ValueError("Beta must be either a single value or of length 'hidden_size'.")
                else:
                    raise TypeError("Beta must be a number or a torch.Tensor.")
                    
    def _init_weights_hi(self, x: float = 1):
        """Initialize the input-to-hidden weights."""
        with torch.no_grad():
            new_weights = torch.eye(self.rnn.hidden_size) * x
            self.rnn.weight_ih_l0.copy_(new_weights)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LeakyKernel.

        Args:
            input_ (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing through the RNN.
        """
        input_ = input_.to(self.device)
        mem, _ = self.rnn(input_)
        
        if self._scale:
            return LeakyKernel.V0 * mem / self._dt
        
        return mem
    
    @staticmethod    
    def assimulate_response(input_seq: SpikeSample, tau: float, dt: float) -> torch.Tensor:
        """
        Simulate the response of the leaky integrator to a given input spike sequence.

        Args:
            input_seq (SpikeSample): Input spike sequence.
            tau (float): Time constant of the leaky integrator.
            dt (float): Time step.

        Returns:
            torch.Tensor: Simulated response of the leaky integrator.
        """
        seq_length = input_seq.seq_len 
        input_size = input_seq.size

        response = torch.zeros(seq_length, input_size)
        times = torch.arange(seq_length, dtype=torch.float32)
        
        for neuron in input_seq.get():
            spikes = neuron.get_spike_times()
            i = neuron.get_index()
            for t in spikes:
                exp = torch.exp(-(times[t:] - t) * dt / tau)
                response[t:, i] += (1 / tau) * exp

        return LeakyKernel.V0 * response