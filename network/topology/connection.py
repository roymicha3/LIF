from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.nn import Module, Parameter

from network.nodes.node import Node


class Connection(ABC, Module):
    """
    Abstract base method for connections between ``Nodes``.
    """

    def __init__(
        self,
        source: Node,
        target: Node,
        device=None
    ) -> None:
        super().__init__()
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

        mask = kwargs.get("mask", None)
        if mask is not None:
            self.w.masked_fill_(mask, 0)

    @abstractmethod
    def reset_state_variables(self) -> None:
        """
        Reset logic for the connection.
        """

