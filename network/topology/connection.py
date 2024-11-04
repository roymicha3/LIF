import torch
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from abc import ABC, abstractmethod
from torch.nn import Module, Parameter

class Connection(ABC, Module):
    """
    Abstract base class for connections between ``Nodes`` in a neural network.
    
    This class provides a foundation for implementing various types of connections
    between neurons or layers in a network.
    """

    def __init__(
        self,
        w: Optional[torch.Tensor] = None,
        dim: Optional[Tuple[int, int]] = None,
        device: Optional[str] = None
    ) -> None:
        """
        Initialize a Connection.

        :param w: Pre-defined weight tensor. If None, weights will be sampled.
        :param dim: Dimensions of the weight matrix (input_size, output_size).
        :param device: Device to store the weights (CPU or GPU).
        """
        super().__init__()
        
        # Ensure that either weights or dimensions are provided, but not both
        assert (w is None) != (dim is None), "Either weights or dimensions must be provided, but not both."
        
        if w is None:
            assert dim is not None, "Dimensions must be provided if weights are not."
            self.w = Parameter(self.sample_weights(dim[0], dim[1], device), requires_grad=True)
        else:
            self.w = Parameter(w, requires_grad=True)
        
        self.device = device
        
    @property
    def size(self) -> Tuple[int, int]:
        """
        Return the size of the connection weight matrix.

        :return: A tuple containing the input and output dimensions of the connection.
        """
        return self.w.size()

    @abstractmethod
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Compute pre-activations of downstream neurons given spikes of upstream neurons.

        :param input_: Incoming spikes.
        :return: Pre-activations of downstream neurons.
        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Compute connection's update rule.

        :param bool learning: Whether to allow connection updates.
        :param ByteTensor mask: Boolean mask determining which weights to clamp to zero.
        """
        learning = kwargs.get("learning", True)

        if learning:
            self.update_rule.update(**kwargs)

        mask = kwargs.get("mask", None)
        if mask is not None:
            self.w.masked_fill_(mask, 0)

    @abstractmethod
    def reset_state_variables(self) -> None:
        """
        Reset internal state variables of the connection.
        """
        pass
        
    @staticmethod
    def sample_weights(input_size: int, output_size: int, device: Optional[str] = None) -> torch.Tensor:
        """
        Sample weights from a normal distribution with mean 0.5 and standard deviation 1.

        :param input_size: Number of input neurons.
        :param output_size: Number of output neurons.
        :param device: Device to store the weights (CPU or GPU).
        :return: A tensor of sampled weights.
        """
        return torch.normal(0.5, 1.0, size=(input_size, output_size), device=device)

    def plot_weights_histogram(self, bins: int = 25) -> None:
        """
        Plot a histogram of the weight values.
        
        :param bins: Number of bins to use for the histogram.
        """
        weights = self.w.detach().cpu().numpy()
        
        plt.figure(figsize=(8, 6))
        plt.hist(weights.flatten(), bins=bins, color='blue', alpha=0.7)
        plt.title("Histogram of Weight Values")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()
        
    def partial_forward(self, input_: torch.Tensor, k: int) -> torch.Tensor:
        """
        Compute the output for only the k-th index of the result.

        :param input_: Incoming spikes. Shape: (batch, times, n)
        :param k: Index of the output neuron to compute
        :return: Output for the k-th neuron. Shape: (batch, times, 1)
        """
        assert 0 <= k < self.w.size(1), f"k must be between 0 and {self.w.size(1) - 1}"
        
        # Compute the dot product of input with only the k-th column of weights
        result = torch.einsum('bti,i->bt', input_, self.w[:, k])
        
        # Reshape to (batch, times, 1)
        return result.unsqueeze(-1)