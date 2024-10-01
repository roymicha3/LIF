from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
from torch.nn.modules.utils import _pair, _triple

from nodes.node import Node


class AbstractConnection(ABC, Module):
    # language=rst
    """
    Abstract base method for connections between ``Nodes``.
    """

    def __init__(
        self,
        source: Node,
        target: Node,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for abstract base class for connection objects.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
         :param nu: Learning rate for both pre- and post-synaptic events. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param Union[float, torch.Tensor] wmin: Minimum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param Union[float, torch.Tensor] wmax: Maximum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param float norm: Total weight per target neuron normalization.
        :param Union[bool, torch.Tensor] Dales_rule: Whether to enforce Dale's rule. input is boolean tensor in weight shape
            where True means force zero or positive values and False means force zero or negative values.
        """
        super().__init__()

        self.source = source
        self.target = target

        self.weight_decay = weight_decay
        self.reduction = reduction

        from ..learning import NoOp

        self.update_rule = kwargs.get("update_rule", None)

        # Float32 necessary for comparisons with +/-inf
        self.wmin = Parameter(
            torch.as_tensor(kwargs.get("wmin", -np.inf), dtype=torch.float32),
            requires_grad=False,
        )
        self.wmax = Parameter(
            torch.as_tensor(kwargs.get("wmax", np.inf), dtype=torch.float32),
            requires_grad=False,
        )
        self.norm = kwargs.get("norm", None)
        self.decay = kwargs.get("decay", None)

        if self.update_rule is None:
            self.update_rule = NoOp

        self.update_rule = self.update_rule(
            connection=self,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs,
        )

        self.Dales_rule = kwargs.get("Dales_rule", None)
        if self.Dales_rule is not None:
            self.Dales_rule = Parameter(
                torch.as_tensor(self.Dales_rule, dtype=torch.bool), requires_grad=False
            )

    @abstractmethod
    def compute(self, s: torch.Tensor) -> None:
        # language=rst
        """
        Compute pre-activations of downstream neurons given spikes of upstream neurons.

        :param s: Incoming spikes.
        """

    @abstractmethod
    def update(self, **kwargs) -> None:
        # language=rst
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

        if self.Dales_rule is not None:
            # weight that are negative and should be positive are set to 0
            self.w[self.w < 0 * self.Dales_rule.to(torch.float)] = 0
            # weight that are positive and should be negative are set to 0
            self.w[self.w > 0 * 1 - self.Dales_rule.to(torch.float)] = 0

    @abstractmethod
    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """


class Connection(AbstractConnection):
    # language=rst
    """
    Specifies synapses between one or two populations of neurons.
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a :code:`Connection` object.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
         :param nu: Learning rate for both pre- and post-synaptic events. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param Union[float, torch.Tensor] wmin: Minimum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param Union[float, torch.Tensor] wmax: Minimum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)

        w = kwargs.get("w", None)
        if w is None:
            if (self.wmin == -np.inf).any() or (self.wmax == np.inf).any():
                w = torch.clamp(torch.rand(source.n, target.n), self.wmin, self.wmax)
            else:
                w = self.wmin + torch.rand(source.n, target.n) * (self.wmax - self.wmin)
        else:
            if (self.wmin != -np.inf).any() or (self.wmax != np.inf).any():
                w = torch.clamp(torch.as_tensor(w), self.wmin, self.wmax)

        self.w = Parameter(w, requires_grad=False)

        b = kwargs.get("b", None)
        if b is not None:
            self.b = Parameter(b, requires_grad=False)
        else:
            self.b = None

        if isinstance(self.target, CSRMNodes):
            self.s_w = None

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute pre-activations given spikes using connection weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
                 decaying spike activation).
        """
        # Compute multiplication of spike activations by weights and add bias.
        if self.b is None:
            post = s.view(s.size(0), -1).float() @ self.w
        else:
            post = s.view(s.size(0), -1).float() @ self.w + self.b
        return post.view(s.size(0), *self.target.shape)

    def compute_window(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """ """

        if self.s_w == None:
            # Construct a matrix of shape batch size * window size * dimension of layer
            self.s_w = torch.zeros(
                self.target.batch_size, self.target.res_window_size, *self.source.shape
            )

        # Add the spike vector into the first in first out matrix of windowed (res) spike trains
        self.s_w = torch.cat((self.s_w[:, 1:, :], s[:, None, :]), 1)

        # Compute multiplication of spike activations by weights and add bias.
        if self.b is None:
            post = (
                self.s_w.view(self.s_w.size(0), self.s_w.size(1), -1).float() @ self.w
            )
        else:
            post = (
                self.s_w.view(self.s_w.size(0), self.s_w.size(1), -1).float() @ self.w
                + self.b
            )

        return post.view(
            self.s_w.size(0), self.target.res_window_size, *self.target.shape
        )

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)

    def normalize(self) -> None:
        # language=rst
        """
        Normalize weights so each target neuron has sum of connection weights equal to
        ``self.norm``.
        """
        if self.norm is not None:
            w_abs_sum = self.w.abs().sum(0).unsqueeze(0)
            w_abs_sum[w_abs_sum == 0] = 1.0
            self.w *= self.norm / w_abs_sum

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()
