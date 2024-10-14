from abc import ABC, abstractmethod
import numpy as np
import torch
from torch.nn import Module, Parameter

from network.nodes.node import Node

from network.learning.learning_rule import MaxTimeConnectionGrad


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
    def forward(self, input_: torch.Tensor) -> None:
        """
        Compute pre-activations of downstream neurons given spikes of upstream neurons.

        :param input_: Incoming spikes.
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

        self.w = Parameter(w, requires_grad=True)

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Compute pre-activations given spikes using connection weights.

        :param input_: Incoming spikes of shape (batch_size, n_inputs) or (n_inputs,).
        :return: Incoming spikes multiplied by synaptic weights of shape (batch_size, n_outputs) or (n_outputs,).
        """
        # Check if input is a single sample or a batch
        if input_.dim() == 1:  # Single sample
            input_ = input_.unsqueeze(0)  # Add a batch dimension

        output = input_ @ self.w  # Matrix multiplication between input spikes and weights
        
        # Save the tensors for the backward pass
        self.saved_tensors = (input_, output)
        return output

    def backward(self, output_grad: torch.Tensor, max_idx: torch.Tensor) -> tuple:
        """
        Backward function for the learning rule.
        Computes the gradient of the loss with respect to inputs and weights.

        :param output_grad: Gradient of the loss with respect to the output, shape (batch_size, n_outputs) or (n_outputs,).
        :param max_idx: Indices of the selected maximum spikes.
        :return: Gradients with respect to the input and weights.
        """
        input_, output = self.saved_tensors
        
        # Check if input is a single sample or a batch
        if input_.dim() == 1:  # Single sample
            input_ = input_.unsqueeze(0)  # Add a batch dimension if necessary

        # Use advanced indexing to select the input values based on max_idx
        input_values = input_.gather(dim=0, index=max_idx.unsqueeze(-1).expand(-1, -1, input_.size(1)))

        # Compute the gradient of the input
        grad_input = output_grad @ self.w  # Backpropagate through weights
        grad_weight = output_grad.t().mm(input_values.view(-1, input_values.size(-1)))  # Reshape for correct multiplication

        return grad_input, grad_weight


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
