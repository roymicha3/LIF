import torch
import matplotlib.pyplot as plt
from typing import List
from network.kernel.kernel import Kernel
from network.topology.connection import Connection
from network.neuron.neuron import Neuron, NeuronOutputType
from data.spike.spike_data import SpikeData
from data.spike.spike_sample import SpikeSample
from common import SPIKE_NS
from network.learning.grad_wrapper import GradWrapper, ConnectionGradWrapper

class SingleSpikeNeuron(Neuron):
    """
    A neuron model that processes spike inputs and produces a single spike output.
    This neuron computes an initial voltage, determines spike times, and then
    recomputes the voltage based on these spike times.
    """

    def __init__(self,
                config,
                kernel: Kernel,
                connection: Connection,
                threshold: float = 1.0,
                type_: NeuronOutputType = NeuronOutputType.SPIKE):
        """
        Initialize the SingleSpikeNeuron.

        Args:
            kernel (Node): The kernel used for processing input spikes.
            connection (Connection): The connection object for signal propagation.
            threshold (float): The voltage threshold for spike generation. Default is 1.0.
        """
        super().__init__(config, type_)
        self._kernel = kernel
        self._connection = connection
        self._threshold = threshold if threshold else config[SPIKE_NS.v_thr]
        
        self.saved_tensors = None
        
    def forward(self, input_spikes: List[SpikeSample]):
        """
        Perform the forward pass of the neuron.

        Args:
            input_spikes: The input spike data.

        Returns:
            The result of the forward pass.
        """
        self.saved_tensors = [], [], []
        res = []
        for sample in input_spikes:
            current_res = self.single_forward(sample)
            res.append(current_res)
        
        input_voltage, final_voltage, spike_times = self.saved_tensors
        
        self.saved_tensors = torch.stack(input_voltage, dim=0), torch.stack(final_voltage, dim=0), torch.stack(spike_times, dim=0)
        
        if self._type is NeuronOutputType.VALUE:
            return torch.tensor(res)
            
        return res
    
    def single_forward(self, input_spikes: SpikeSample):
        input_spikes_tensor = input_spikes.to_torch()
        
        input_voltage = self._kernel.forward(input_spikes_tensor)
        initial_voltage = self._connection.forward(input_voltage)
        
        spike_times = self.first_spike_index(initial_voltage, self._threshold)
        final_voltage = self._recompute_voltage(input_spikes, spike_times)
        
        self.saved_tensors[0].append(input_voltage)
        self.saved_tensors[1].append(final_voltage)
        self.saved_tensors[2].append(spike_times)
        
        if self._type is NeuronOutputType.VALUE:
            return torch.max(final_voltage, dim=-2)[0] - self._threshold
            
        data = []
        for neuron_idx in range(self._connection.size[1]):
            spike = spike_times[neuron_idx]
            if spike == 0:
                continue
            
            data.append(
                SpikeData(self._config, neuron_idx, [spike * self._config[SPIKE_NS.dt]]))
        
        return SpikeSample(self._config, data, input_spikes.get_label())
    
    def backward(self, output_grad: GradWrapper) -> GradWrapper:
        """
        Perform the backward pass of the neuron.

        Args:
            output_grad: The gradient of the output.

        Returns:
            The result of the backward pass.
        """
        input_voltage, final_voltage, spike_times = self.saved_tensors
        res = torch.max(final_voltage, dim=-2)
        
        indices = res.indices
        voltage = res.values
        
        weight_grads = []
        for input_, grad, index in zip(input_voltage, output_grad.grad, indices):
            weight_grads.append(torch.mm(grad, input_[index, :]).t())
        
        grad = ConnectionGradWrapper(output_grad.grad, torch.stack(weight_grads, dim=0))
        input_grad = self._connection.backward(grad)
        
        return GradWrapper(input_grad)
        

    def _compute_initial_voltage(self, input_spikes: SpikeSample) -> torch.Tensor:
        """
        Compute the initial voltage from input spikes.

        Args:
            input_spikes (SpikeSample): The input spike sample.

        Returns:
            torch.Tensor: The initial voltage.
        """
        spike_tensor = input_spikes.to_torch()
        input_voltage = self._kernel.forward(spike_tensor)
        return self._connection.forward(input_voltage)

    def _recompute_voltage(self, input_spikes: SpikeSample, spike_times: torch.Tensor) -> torch.Tensor:
        """
        Recompute voltage considering the first spike times.

        Args:
            input_spikes (SpikeSample): The input spike sample.
            spike_times (torch.Tensor): The times of the first spikes.

        Returns:
            torch.Tensor: The recomputed voltage.
        """
        output_size = self._connection.size[1]
        voltage = torch.zeros((input_spikes.seq_len, output_size), dtype=torch.float)

        for k in range(output_size):
            silenced_spikes = input_spikes.silence(spike_times[k])
            voltage[:, k] = self._compute_neuron_voltage(silenced_spikes, k).squeeze()

        return voltage

    def _compute_neuron_voltage(self, input_spikes: SpikeSample, neuron_index: int) -> torch.Tensor:
        """
        Compute voltage for a single neuron.

        Args:
            input_spikes (SpikeSample): The input spike sample.
            neuron_index (int): The index of the neuron.

        Returns:
            torch.Tensor: The computed voltage for the neuron.
        """
        spike_tensor = input_spikes.to_torch()
        input_voltage = self._kernel.forward(spike_tensor)
        return self._connection.partial_forward(input_voltage, neuron_index)

    @staticmethod
    def first_spike_index(voltage: torch.Tensor, threshold: float) -> torch.Tensor:
        """
        Find the index of the first spike for each neuron.

        Args:
            voltage (torch.Tensor): The voltage tensor.
            threshold (float): The threshold for spike detection.

        Returns:
            torch.Tensor: The indices of the first spikes.
        """
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
        plt.plot(time_steps, initial_voltage.squeeze().detach().numpy(), label='Initial Voltage', linestyle='--')
        plt.plot(time_steps, final_voltage.squeeze().detach().numpy(), label='Final Voltage')
        
        plt.axhline(y=self._threshold, color='r', linestyle='-', label='Threshold')
        
        plt.xlabel('Time Steps')
        plt.ylabel('Voltage')
        plt.title('Initial and Final Voltages with Threshold')
        plt.legend()
        plt.grid(True)
        plt.show()