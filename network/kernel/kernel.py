from functools import reduce
from operator import mul
from typing import Iterable, Optional

import torch


class Kernel(torch.nn.Module):
    """
    Base class for groups of neurons.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        learning: bool = False
    ) -> None:
        """
        Base class constructor.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param learning: Whether to be in learning or testing.
        """
        super().__init__()

        assert (
            n is not None or shape is not None
        ), "Must provide either no. of neurons or shape of layer"

        if n is None:
            self.n = reduce(mul, shape)  # No. of neurons product of shape.
        else:
            self.n = n  # No. of neurons provided.

        if shape is None:
            self.shape = [self.n]  # Shape is equal to the size of the layer.
        else:
            self.shape = shape  # Shape is passed in as an argument.

        self.learning = learning

    def forward(self, input_: torch.Tensor) -> None:
        """
        Base class method for a single simulation step.

        :param input_: Inputs to the layer.
        """
        return input_
    
    def backward(self, E: torch.Tensor) -> torch.Tensor:
        """
        Base class method for backpropagation.

        :param E: Error tensor.
        """
        return E

    def train(self, mode: bool = True) -> "Node":
        """
        Sets the layer in training mode.

        :param bool mode: Turn training on or off
        :return: self as specified in `torch.nn.Module`
        """
        self.learning = mode
        return super().train(mode)