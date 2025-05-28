from typing import List

import numpy as np
import torch
from experiment_manager.environment import Environment

from network.learning.learning_rule import LearningRule
from network.topology.connection import Connection


class SimpleConnection(Connection):
    """
    Specifies synapses between one or two populations of neurons
    """

    def __init__(
                self,
                lr_list: List[LearningRule],
                env: Environment,
                input_size: int = None,
                output_size: int = None,
                w: torch.Tensor = None,
                device=None,
                norm: np.int32 = 1) -> None:
        
        super().__init__(lr_list, (input_size, output_size), w, device)
        self.env = env
        self.norm = norm
        self.saved_tensors = None
        
        self.batch_norm = torch


    def partial_forward(self, input_: torch.Tensor) -> torch.Tensor:
        input_ = input_.to(self.device)
        
        if input_.dim() == 1:  # Single sample
            input_ = input_.unsqueeze(0)  # Add a batch dimension

        output = input_ @ self.w  # Matrix multiplication between input spikes and weights
        return output
    
    
    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        """
        Compute pre-activations given spikes using connection weights and bias.

        :param input_: Incoming spikes of shape (batch_size, n_inputs) or (n_inputs,).
        :return: Incoming spikes multiplied by synaptic weights and bias.
        """
        output = self.partial_forward(input_) # TODO: the logic here is wrong for sequential data
        
        spikes = torch.zeros_like(output, device=self.device)  # Initialize spikes tensor
        
        for lr in self.lr_list:
            spikes += lr.forward(input_, output) # Forward pass of the learning rule

        if torch.is_grad_enabled():
            self.saved_tensors = input_, output # Save for backward pass
        
        return output, spikes

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

        batch_size = input_.size(0)
        weight_grad = torch.zeros(
            shape=(batch_size, self.w.size(0), self.w.size(1)),
            device=self.device)
        
        for lr in self.lr_list:
            weight_grad += lr.backward(input_, grad)
        
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

        self.w.grad = grad.to(self.device)

        # Monitor values
        with torch.no_grad():
            if torch.isnan(self.w.grad).any():
                self.env.logger.error("NaN in weight gradients")
            if torch.isinf(self.w.grad).any():
                self.env.logger.error("Inf in weight gradients")
            if self.w.grad.max() > 1e3 or self.w.grad.min() < -1e3:
                self.env.logger.warning("Large values in weight gradients")
                
            if self.w.grad.max() < 1e-3 and self.w.grad.min() > -1e-3:
                self.env.logger.info("\nSmall values in weight gradients \n")

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
        
    def reset(self) -> None:
        """
        Reset the connection.
        """
        self.learning_rule.reset()

