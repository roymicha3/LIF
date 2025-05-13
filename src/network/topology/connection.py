from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import torch
from torch.nn import Module, Parameter

from network.learning.learning_rule import LearningRule


class Connection(ABC, Module):
    """
    Abstract base method for connections,
    This class incorporates the activation inside the connection!
    """

    def __init__(self, lr: LearningRule, shape: tuple = None, w: torch.Tensor = None, device=None) -> None:
        super().__init__()
        self.device = device
        self.learning_rule = lr
        
        if w is not None:
            if not isinstance(w, torch.Tensor):
                raise TypeError("w must be a torch.Tensor")
            self.w = Parameter(w, requires_grad=True)
        elif shape is not None:
            self.w = Parameter(Connection.sample_weights(*shape, device), requires_grad=True)
        else:
            raise ValueError("Either n or w must be provided")

    @abstractmethod
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Compute pre-activations of downstream neurons given spikes of upstream neurons.

        :param input_: Incoming spikes.
        """
        raise NotImplementedError
    
    @abstractmethod
    def partial_forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Compute the inner state of the connection
        """
        raise NotImplemented

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Compute connection's update rule.

        :param bool learning: Whether to allow connection updates.
        """
        learning = kwargs.get("learning", True)

        if learning:
            self.update_rule.update(**kwargs)

    @abstractmethod
    def reset_state_variables(self) -> None:
        """
        Reset logic for the connection.
        """
        
    @staticmethod
    def sample_weights(input_size: int, output_size: int, device=None) -> torch.Tensor:
        """
        Samples weights from a normal distribution with mean 0.0 and standard deviation 1.

        :param input_size: Number of inputs.
        :param output_size: Number of outputs.
        :param device: Device to store the weights (CPU or GPU).
        :return: A tensor of sampled weights.
        """
        # Sample from Norm(0.0, 1)
        weights = torch.normal(0.0, 1.0, size=(input_size, output_size), device=device)
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