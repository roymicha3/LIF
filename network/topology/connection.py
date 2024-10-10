from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.nn import Module, Parameter

from network.nodes.node import Node


class AbstractConnection(ABC, Module):
    # language=rst
    """
    Abstract base method for connections between ``Nodes``.
    """

    def __init__(
        self,
        source: Node,
        target: Node
        ) -> None:
        super().__init__()
        self.source = source
        self.target = target

    @abstractmethod
    def forward(self, s: torch.Tensor) -> None:
        """
        Compute pre-activations of downstream neurons given spikes of upstream neurons.

        :param s: Incoming spikes.
        """

    @abstractmethod
    def update(self, **kwargs) -> None:
        """
        Compute connection's update rule.

        Keyword arguments:

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
        # language=rst
        """
        Contains resetting logic for the connection.
        """


class Connection(AbstractConnection):
    """
    Specifies synapses between one or two populations of neurons.
    """

    def __init__(
        self,
        source: Node,
        target: Node,
        w: torch.Tensor = None,
        b: torch.Tensor = None,
        wmin: np.int32 = -np.inf,
        wmax: np.int32 = np.inf,
        norm: np.int32 = None
    ) -> None:
        """
        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        """
        super().__init__(source, target)
        self.w = w
        self.b = b
        
        if self.b is None:
            self.b = torch.zeros(target.n)
        
        assert self.b.size()[0] == target.n
        
        self.wmin = wmin
        self.wmax = wmax
        self.norm = norm
        
        # set w to random values
        if self.w is None:
            if self.wmin == -np.inf or self.wmax == np.inf:
                w = torch.clamp(torch.rand(source.n, target.n), self.wmin, self.wmax)
            else:
                w = self.wmin + torch.rand(source.n, target.n) * (self.wmax - self.wmin)
        else:
            if (self.wmin != -np.inf).any() or (self.wmax != np.inf).any():
                w = torch.clamp(torch.as_tensor(w), self.wmin, self.wmax)

        self.w = Parameter(w, requires_grad=False)

        if b is not None:
            self.b = Parameter(b, requires_grad=False)
        else:
            self.b = None

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """
        Compute pre-activations given spikes using connection weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights.
        """
        if self.b is None:
            post = s.view(s.size(0), -1).float() @ self.w
        else:
            post = s.view(s.size(0), -1).float() @ self.w + self.b
        
        return post.view(s.size(0), *self.target.shape)


    def update(self, **kwargs) -> None:
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)

    def normalize(self) -> None:
        """
        Normalize weights so each target neuron has sum of connection weights equal to
        ``self.norm``.
        """
        if self.norm is not None:
            w_abs_sum = self.w.abs().sum(0).unsqueeze(0)
            w_abs_sum[w_abs_sum == 0] = 1.0
            self.w *= self.norm / w_abs_sum

    def reset_state_variables(self) -> None:
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()
