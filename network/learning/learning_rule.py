import warnings
from abc import ABC
from typing import Optional, Sequence, Union
import torch

class MaxTimeConnectionGrad(torch.autograd.Function):
    """
    Max time learning rule (simple learning rule for fully connected linear layer).
    """

    @staticmethod
    def forward(ctx, input_, weight_):
        """
        Forward function for the learning rule.
        Computes the linear transformation and saves variables for backward pass.
        """
        output = input_ @ weight_
        max_val, max_idx = torch.max(output, dim=-1)  # Max along the last dimension
        
        # Save the tensors for the backward pass
        ctx.save_for_backward(input_, weight_, max_idx)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward function for the learning rule.
        Computes the gradient of the loss with respect to inputs and weights.
        """
        # Retrieve saved tensors from the context
        input_, weight_, max_idx = ctx.saved_tensors
        
        grad_output = grad_output.sum(dim=1)
        
        # Compute the gradient of the input
        grad_input = grad_output @ weight_  # Gradient w.r.t. input #TODO: fix the shape of the grad_output!!!
        # Use `max_idx` to compute the correct gradient for the weights
        grad_weight = grad_output.t().mm(input_.index_select(0, max_idx))  # Select input based on max_idx

        return grad_input, grad_weight
