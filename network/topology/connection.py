from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import Module, Parameter

from network.nodes.node import Node


class Connection(ABC, Module):
    """
    Abstract base method for connections between ``Nodes``.
    """

    def __init__(
        self,
        w: torch.Tensor,
        source: Node,
        target: Node,
        device=None
    ) -> None:
        super().__init__()
        self.w = w
        
        if self.w is None:
            self.w = Parameter(Connection.sample_weights(source.n, target.n, device), requires_grad=True)
        
        self.source = source
        self.target = target
        self.device = device

    @abstractmethod
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Compute pre-activations of downstream neurons given spikes of upstream neurons.

        :param input_: Incoming spikes.
        """

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

        # mask = kwargs.get("mask", None)
        # if mask is not None:
        #     self.w.masked_fill_(mask, 0)

    @abstractmethod
    def reset_state_variables(self) -> None:
        """
        Reset logic for the connection.
        """
        
    @staticmethod
    def sample_weights(input_size: int, output_size: int, device=None) -> torch.Tensor:
        """
        Samples weights from a normal distribution with mean 0.5 and standard deviation 1.

        :param input_size: Number of inputs.
        :param output_size: Number of outputs.
        :param device: Device to store the weights (CPU or GPU).
        :return: A tensor of sampled weights.
        """
        # Sample from Norm(0.5, 1)
        weights = torch.normal(0.5, 1.0, size=(input_size, output_size), device=device)
        return weights

    def plot_weights_histogram(self, bins=25):
        """
        Plot a histogram of the weight values.
        
        :param bins: Number of bins to use for the histogram (default is 30).
        """
        # Convert the weights to a numpy array
        weights = self.w.detach().cpu().numpy()
        
        # Plot the histogram
        plt.figure(figsize=(8, 6))
        plt.hist(weights.flatten(), bins=bins, color='blue', alpha=0.7)
        plt.title("Histogram of Weight Values")
        plt.xlabel("Weight Value")
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.show()