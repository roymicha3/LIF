import numpy as np
import torch
from torch.nn import Module, Parameter


from network.nodes.node import Node
from network.learning.grad_wrapper import GradWrapper, ConnectionGradWrapper
from network.topology.connection import Connection

class SimpleConnection(Connection):
    """
    Specifies synapses between one or two populations of neurons, with optional bias.
    """

    def __init__(
        self,
        source: Node,
        target: Node,
        norm: np.int32 = 1,
        device=None
    ) -> None:
        """
        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param bias: Whether to include a bias term in the connection.
        """
        super().__init__(SimpleConnection.sample_weights(source.n, target.n, device),
                         source,
                         target,
                         device=device)
        self.norm = norm
        self.saved_tensors = None

        self.w = Parameter(self.w, requires_grad=True)

    @staticmethod
    def sample_weights(input_size: int, output_size: int, device=None) -> torch.Tensor:
        """
        Samples weights from a normal distribution with mean 0.5 and standard deviation 1.

        :param input_size: Number of inputs.
        :param output_size: Number of outputs.
        :param device: Device to store the weights (CPU or GPU).
        :return: A tensor of sampled weights.
        """
        # Sample from Norm(0.5, 1)
        weights = torch.normal(0.5, 1.0, size=(input_size, output_size), device=device)
        return weights

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

        self.saved_tensors = input_  # Save for backward pass
        return output

    def backward(self, output_grad: GradWrapper) -> tuple:
        """
        Backward function for the learning rule.
        Computes the gradient of the loss with respect to inputs, weights, and bias.

        :param output_grad: Gradient of the loss with respect to the output.
        :return: Gradients with respect to the input, weights, and bias.
        """
        input_ = self.saved_tensors
        
        max_idx = output_grad.info["max_idx"]
        grad = output_grad.output_grad.to(self.device)
        
        # Check if input is a single sample or a batch
        if input_.dim() == 1:  # Single sample
            input_ = input_.unsqueeze(0)  # Add a batch dimension if necessary

        res = []
        
        # plasticity_induction = output_grad.info["plasticity_induction"]
        
        # enumerating over batch data
        for i, idx in enumerate(max_idx):
            res.append((grad[i] @ input_[i, idx, :]).t())
            
        grad_weight = torch.stack(res)
        
        # Compute the gradient of the input
        grad_input = grad @ self.w.t()  # Backpropagate through weights

        total_grad = ConnectionGradWrapper(grad_input, grad_weight)
        self.update(total_grad)  # Update weights and bias

        return total_grad

    def update(self, grad: ConnectionGradWrapper) -> None:
        """
        Update weights and bias based on gradients.
        """
        batch_size = grad.weight_grad.size(0)
        if grad.weight_grad.dim() > self.w.dim():
            grad.weight_grad = torch.sum(grad.weight_grad, dim=0) / batch_size

        self.w.grad = grad.weight_grad

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

