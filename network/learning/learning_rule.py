import torch
from typing import Tuple
from abc import ABC, abstractmethod


class LearningRule(ABC, torch.nn.Module):
    """
    Abstract base class for learning rules in Spiking Neural Networks (SNNs) using PyTorch.
    """

    @abstractmethod
    def backward(self, input_, E, **kwargs):
        """
        Backward function for the learning rule.
        """
        pass

    
    @abstractmethod
    def forward(self, input_data, **kwargs):
        """
        Forward pass of the learning rule.
        """
        pass