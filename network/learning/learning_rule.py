import torch
from typing import Tuple
from abc import ABC, abstractmethod


class LearningRule(ABC, torch.nn.Module):
    """
    Abstract base class for learning rules in Spiking Neural Networks (SNNs) using PyTorch.
    """

    @abstractmethod
    def backward(self, input_data, output_data, E, **kwargs) -> Tuple(torch.Tensor, torch.Tensor):
        """
        Update synaptic weights based on pre- and post-synaptic spikes.

        Args:
            pre_spikes (torch.Tensor): Spike times or spike trains of pre-synaptic neurons.
            post_spikes (torch.Tensor): Spike times or spike trains of post-synaptic neurons.
            weights (torch.Tensor): Current synaptic weights.
            **kwargs: Additional parameters specific to the learning rule.

        Returns:
            torch.Tensor: Updated synaptic weights.
        """
        pass

    
    @abstractmethod
    def forward(self, input_data, **kwargs):
        """
        Forward pass of the learning rule.
        """
        pass