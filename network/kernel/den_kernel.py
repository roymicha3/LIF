import torch
from network.kernel.kernel import Kernel
from network.kernel.leaky_kernel import LeakyKernel
from common import Configuration, SPIKE_NS
from data.spike.spike_sample import SpikeSample

class DENKernel(Kernel):
    """
    Dual Exponential Neuron (DEN) Kernel.
    This class implements a neuron model with separate time constants for
    membrane potential and synaptic conductance.
    """

    def __init__(
        self,
        config: Configuration,
        n: int,
        device=None,
        dtype=None,
        scale: bool = False,
        learning: bool = False
    ):
        """
        Initialize the DENKernel.

        Args:
            config (Configuration): Configuration object containing model parameters.
            n (int): Number of neurons in the layer.
            device: The device to run the model on (e.g., 'cpu' or 'cuda').
            dtype: The data type for the model parameters.
            scale (bool): Whether to scale the output. Default is False.
            learning (bool): Whether the model is in learning mode. Default is False.
        """
        super(DENKernel, self).__init__(n, learning=learning)
        
        self._device = device
        self._config = config
        tau_m = self._config[SPIKE_NS.tau_m]
        self._conductance = LeakyKernel(self._config, n, device, dtype, scale=scale, learning=learning, tau=tau_m)
        
        tau_s = self._config[SPIKE_NS.tau_s]  # TODO: Determine appropriate usage of tau_s
        self._voltage = LeakyKernel(self._config, n, device, dtype, scale=False, learning=False, tau=tau_m / 4)  # TODO: Confirm this tau value
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DENKernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after processing through the conductance and voltage kernels.
        """
        with torch.no_grad():
            x = x.to(self._device)
            mem = self._conductance(x)
            mem = self._voltage(mem)
        
        return mem
    
    @staticmethod    
    def simulate_response(input_seq: SpikeSample, tau_m: float, tau_s: float, dt: float) -> torch.Tensor:
        """
        Simulate the response of the DEN model to a given input spike sequence.

        Args:
            input_seq (SpikeSample): Input spike sequence.
            tau_m (float): Membrane time constant.
            tau_s (float): Synaptic time constant.
            dt (float): Time step.

        Returns:
            torch.Tensor: Simulated response of the DEN model.
        """
        seq_length = input_seq.seq_len
        input_size = input_seq.size

        response = torch.zeros(seq_length, input_size)
        times = torch.arange(seq_length, dtype=torch.float32)
        
        for neuron in input_seq.get():
            spikes = neuron.get_spike_times()
            i = neuron.get_index()
            for t in spikes:
                exp_rise = torch.exp(-(times[t:] - t) * dt / tau_m)
                exp_decay = torch.exp(-(times[t:] - t) * dt / tau_s)
                response[t:, i] += (1 / (tau_m - tau_s)) * (exp_rise - exp_decay)

        return response