from functools import reduce
from operator import mul
from typing import Iterable, Optional

import torch


class Kernel(torch.nn.Module):
    """
    Base class for groups of neurons in a spiking neural network.
    This class provides the fundamental structure for creating layers of neurons.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        learning: bool = True
    ) -> None:
        """
        Initialize the Kernel.

        Args:
            n (Optional[int]): The total number of neurons in the layer. If not provided, it will be calculated from the shape.
            shape (Optional[Iterable[int]]): The dimensionality of the layer. If not provided, it will be set to [n].
            learning (bool): Whether the layer is in learning mode (True) or inference mode (False). Defaults to True.

        Raises:
            AssertionError: If neither n nor shape is provided.
        """
        super().__init__()

        assert (
            n is not None or shape is not None
        ), "Must provide either number of neurons (n) or shape of layer"

        if n is None:
            self.n = reduce(mul, shape)  # Calculate total number of neurons from shape
        else:
            self.n = n

        if shape is None:
            self.shape = [self.n]  # Default shape is a 1D array of size n
        else:
            self.shape = list(shape)  # Convert shape to a list for consistency

        # Register a buffer to store spike occurrences
        self.register_buffer("s", torch.ByteTensor())

        self.learning = learning

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Perform a single simulation step (forward pass).

        Args:
            input_ (torch.Tensor): Input tensor to the layer.

        Returns:
            torch.Tensor: The output tensor after processing.
        """
        return input_

    def reset_state_variables(self) -> None:
        """
        Reset the state variables of the layer.
        This method should be called at the beginning of each new simulation.
        """
        self.s.zero_()  # Reset spike occurrences to zero

    def train(self, mode: bool = True) -> "Kernel":
        """
        Set the layer's training mode.

        Args:
            mode (bool): If True, set to training mode; if False, set to evaluation mode.

        Returns:
            Kernel: The Kernel instance (self).
        """
        self.learning = mode
        return super().train(mode)