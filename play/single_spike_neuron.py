import torch
from network.nodes.node import Node
from network.topology.connection import Connection
from play.neuron import Neuron
from data.spike.spike_sample import SpikeSample

class SingleSpikeNeuron(Neuron):
    
    def __init__(self, kernel: Node, connection: Connection, threshold: float = 1.0):
        super().__init__()
        self._kernel = kernel
        self._connection = connection
        self._threshold = threshold
        
    def forward(self, input_spikes):
        return super().forward(input_spikes)
    
    def backward(self, output_grad):
        return super().backward(output_grad)
    
    def _prepare(self, input_spikes: SpikeSample):
        spike_tensor = input_spikes.to_torch()
        input_voltage = self._kernel.forward(spike_tensor)
        v = self._connection.forward(input_voltage)
        output_size = self._connection.size[1]
        
        spike_times = SingleSpikeNeuron.first_spike_index(v, self._threshold)
        
        for k in range(output_size):
            current_input_spikes = input_spikes.silence(spike_times[k])
            current_spike_tensor = current_input_spikes.to_torch()
            current_input_voltage = self._kernel.forward(current_spike_tensor)
            v[:, k] = self._connection.partial_forward(current_input_voltage, k)
        
        return v
            
    @staticmethod
    def first_spike_index(voltage: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Find the index of the first spike (voltage exceeding threshold) for each neuron.

        Args:
        voltage (torch.Tensor): Input voltage tensor. Shape: [n_neurons, time_steps] or [time_steps]
        threshold (float): The voltage threshold for a spike

        Returns:
        torch.Tensor: Indices of the first spike for each neuron. Shape: [n_neurons] or scalar
        """
        # Ensure the input is at least 2D
        if voltage.dim() == 1:
            voltage = voltage.unsqueeze(0)

        # Create a mask where voltages exceed the threshold
        spikes = voltage >= threshold

        # Find the first spike for each neuron
        first_spike = spikes.to(torch.int64).argmax(dim=1)

        # Handle cases where no spike occurred
        no_spike = ~spikes.any(dim=1)
        first_spike[no_spike] = voltage.shape[1]  # Set to time_steps if no spike

        return first_spike.squeeze()  # Remove singleton dimension if input was 1D, might interfere with computations...