import numpy as np
import torch

from network.topology.connection import Connection
from network.learning.learning_rule import LearningRule

class SimpleConnection(Connection):
    """
    Specifies synapses between one or two populations of neurons
    """

    def __init__(
        self,
        lr: LearningRule,
        input_size: int = None,
        output_size: int = None,
        w: torch.Tensor = None,
        device=None,
        norm: np.int32 = 1) -> None:
        
        """
        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param bias: Whether to include a bias term in the connection.
        """
        super().__init__(lr, (input_size, output_size), w, device)
        self.norm = norm
        self.saved_tensors = None


    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Compute pre-activations given spikes using connection weights and bias.

        :param input_: Incoming spikes of shape (batch_size, n_inputs) or (n_inputs,).
        :return: Incoming spikes multiplied by synaptic weights and bias.
        """
        input_ = input_.to(self.device)
        
        if input_.dim() == 1:  # Single sample
            input_ = input_.unsqueeze(0)  # Add a batch dimension

        output = input_ @ self.w  # Matrix multiplication between input spikes and weights
        
        spikes = self.learning_rule.forward(output)  # Forward pass of the learning rule

        if torch.is_grad_enabled():
            self.saved_tensors = input_, output  # Save for backward pass
        
        return spikes

    def backward(self, E):
        """
        Backward function for the learning rule.
        Computes the gradient of the loss with respect to inputs, weights, and bias.

        :param output_grad: Gradient of the loss with respect to the output.
        :return: Gradients with respect to the input, weights, and bias.
        """
        input_, _ = self.saved_tensors
        
        grad = E.to(self.device)
        
        # Check if input is a single sample or a batch
        if input_.dim() == 1:  # Single sample
            input_ = input_.unsqueeze(0)  # Add a batch dimension if necessary

        weight_grad = self.learning_rule.backward(input_, grad)
        
        # Compute the gradient of the input
        input_grad = grad @ self.w.t()  # Backpropagate through weights
        
        self.update(weight_grad)  # Update weights and bias

        return input_grad

    def update(self, grad) -> None:
        """
        Update weights and bias based on gradients.
        """
        batch_size = grad.size(0)
        if grad.dim() > self.w.dim():
            grad = torch.sum(grad, dim=0) / batch_size

        self.w.grad = grad

    def normalize(self) -> None:
        """
        Normalize weights so each target neuron has a sum of connection weights equal to
        ``self.norm``.
        """
        if self.norm is not None:
            w_abs_sum = self.w.abs().sum()
            self.w.divide_(w_abs_sum)

    def reset_state_variables(self) -> None:
        """
        Reset the state variables of the connection.
        """
        super().reset_state_variables()

