import torch
import matplotlib.pyplot as plt
from network.nodes.node import Node
from network.topology.connection import Connection
from play.neuron import Neuron
from data.spike.spike_sample import SpikeSample

class SingleSpikeNeuron(Neuron):
    """
    A neuron model that processes spike inputs and produces a single spike output.
    """

    def __init__(self, kernel: Node, connection: Connection, threshold: float = 1.0):
        """Initialize the SingleSpikeNeuron."""
        super().__init__()
        self._kernel = kernel
        self._connection = connection
        self._threshold = threshold
        
    def forward(self, input_spikes):
        """Forward pass of the neuron."""
        return super().forward(input_spikes)
    
    def backward(self, output_grad):
        """Backward pass of the neuron."""
        return super().backward(output_grad)
    
    def _prepare(self, input_spikes: SpikeSample):
        """Prepare the neuron's response to input spikes."""
        initial_voltage = self._compute_initial_voltage(input_spikes)
        spike_times = self.first_spike_index(initial_voltage, self._threshold)
        final_voltage = self._recompute_voltage(input_spikes, spike_times)
        return final_voltage

    def _compute_initial_voltage(self, input_spikes: SpikeSample) -> torch.Tensor:
        """Compute the initial voltage from input spikes."""
        spike_tensor = input_spikes.to_torch()
        input_voltage = self._kernel.forward(spike_tensor)
        return self._connection.forward(input_voltage)

    def _recompute_voltage(self, input_spikes: SpikeSample, spike_times: torch.Tensor) -> torch.Tensor:
        """Recompute voltage considering the first spike times."""
        output_size = self._connection.size[1]
        voltage = torch.zeros((input_spikes.seq_len, output_size), dtype=torch.float)

        for k in range(output_size):
            silenced_spikes = input_spikes.silence(spike_times[k])
            voltage[:, k] = self._compute_neuron_voltage(silenced_spikes, k).squeeze()

        return voltage

    def _compute_neuron_voltage(self, input_spikes: SpikeSample, neuron_index: int) -> torch.Tensor:
        """Compute voltage for a single neuron."""
        spike_tensor = input_spikes.to_torch()
        input_voltage = self._kernel.forward(spike_tensor)
        return self._connection.partial_forward(input_voltage, neuron_index)

    @staticmethod
    def first_spike_index(voltage: torch.Tensor, threshold: float) -> torch.Tensor:
        """Find the index of the first spike for each neuron."""
        voltage = voltage.unsqueeze(0) if voltage.dim() == 1 else voltage
        spikes = voltage >= threshold
        first_spike = spikes.to(torch.int64).argmax(dim=-2)
        return first_spike
    
    def plot_voltages(self, input_spikes: SpikeSample):
        """
        Plot the initial and final voltages with the threshold marked.
        
        Args:
            input_spikes (SpikeSample): The input spike sample.
        """
        initial_voltage = self._compute_initial_voltage(input_spikes).cpu()
        spike_times = self.first_spike_index(initial_voltage, self._threshold)
        final_voltage = self._recompute_voltage(input_spikes, spike_times).cpu()

        time_steps = range(input_spikes.seq_len)
        
        plt.figure(figsize=(12, 6))
        plt.plot(time_steps, initial_voltage.squeeze().detach().numpy(), label=f'Initial Voltage Neuron', linestyle='--')
        plt.plot(time_steps, final_voltage.squeeze().detach().numpy(), label=f'Final Voltage Neuron')
        
        plt.axhline(y=self._threshold, color='r', linestyle='-', label='Threshold')
        
        plt.xlabel('Time Steps')
        plt.ylabel('Voltage')
        plt.title('Initial and Final Voltages with Threshold')
        plt.legend()
        plt.grid(True)
        plt.show()