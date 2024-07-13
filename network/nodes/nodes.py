from abc import ABC, abstractmethod
from functools import reduce
from operator import mul
from typing import Iterable, Optional, Union

import torch


class Nodes(torch.nn.Module):
    """
    Abstract base class for groups of neurons.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        learning: bool = True,
        **kwargs,
    ) -> None:
        """
        Abstract base class constructor.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record decaying spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
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

        assert self.n == reduce(
            mul, self.shape
        ), "No. of neurons and shape do not match"

        self.traces = traces  # Whether to record synaptic traces.
        self.traces_additive = (
            traces_additive  # Whether to record spike traces additively.
        )
        self.register_buffer("s", torch.ByteTensor())  # Spike occurrences.

        self.sum_input = sum_input  # Whether to sum all inputs.

        if self.traces:
            self.register_buffer("x", torch.Tensor())  # Firing traces.
            self.register_buffer(
                "tc_trace", torch.tensor(tc_trace)
            )  # Time constant of spike trace decay.
            self.register_buffer(
                "trace_scale", torch.tensor(trace_scale)
            )  # Scaling factor for spike trace.
            self.register_buffer(
                "trace_decay", torch.empty_like(self.tc_trace)
            )  # Set in compute_decays.

        if self.sum_input:
            self.register_buffer("summed", torch.FloatTensor())  # Summed inputs.

        self.dt = None
        self.batch_size = None
        self.trace_decay = None
        self.learning = learning

    @abstractmethod
    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Abstract base class method for a single simulation step.

        :param x: Inputs to the layer.
        """
        if self.traces:
            # Decay and set spike traces.
            self.x *= self.trace_decay

            if self.traces_additive:
                self.x += self.trace_scale * self.s.float()
            else:
                self.x.masked_fill_(self.s.bool(), self.trace_scale)

        if self.sum_input:
            # Add current input to running sum.
            self.summed += x.float()

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Abstract base class method for resetting state variables.
        """
        self.s.zero_()

        if self.traces:
            self.x.zero_()  # Spike traces.

        if self.sum_input:
            self.summed.zero_()  # Summed inputs.

    def compute_decays(self, dt) -> None:
        # language=rst
        """
        Abstract base class method for setting decays.
        """
        self.dt = torch.tensor(dt)
        if self.traces:
            self.trace_decay = torch.exp(
                -self.dt / self.tc_trace
            )  # Spike trace decay (per timestep).

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        self.batch_size = batch_size
        self.s = torch.zeros(
            batch_size, *self.shape, device=self.s.device, dtype=torch.bool
        )

        if self.traces:
            self.x = torch.zeros(batch_size, *self.shape, device=self.x.device)

        if self.sum_input:
            self.summed = torch.zeros(
                batch_size, *self.shape, device=self.summed.device
            )

    def train(self, mode: bool = True) -> "Nodes":
        # language=rst
        """
        Sets the layer in training mode.

        :param bool mode: Turn training on or off
        :return: self as specified in `torch.nn.Module`
        """
        self.learning = mode
        return super().train(mode)