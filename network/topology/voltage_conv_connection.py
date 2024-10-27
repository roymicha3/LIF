import numpy as np
import torch


from network.nodes.node import Node
from network.learning.grad_wrapper import GradWrapper, ConnectionGradWrapper
from network.topology.simple_connection import SimpleConnection

class VoltageConvConnection(SimpleConnection):
    """
    Specifies synapses between one or two populations of neurons, with optional bias.
    """

    def __init__(
        self,
        source: Node,
        target: Node,
        norm: np.int32 = 1,
        beta: np.float32 = 1.0,
        device=None
    ) -> None:
        """
        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param bias: Whether to include a bias term in the connection.
        """
        super().__init__(source, target, norm, device=device)
        self.beta = beta


    def backward(self, output_grad: GradWrapper) -> tuple:
        """
        Backward function for the learning rule.
        Computes the gradient of the loss with respect to inputs, weights.

        :param output_grad: Gradient of the loss with respect to the output.
        :return: Gradients with respect to the input, weights.
        """
        #TODO: implement this:
        # w_grad = zeros_like(self.w) -> (n, m)
        # output -> (b, T, n)
        # input -> (b, T, m)
        # grad -> (b, n)
        
        # for i in range(n):
        #     a = exp(beta * output[:,:,i]) ->(b, T)
        #     l = sum(a, dim=1) -> (b)
        #     res = grad[:, i] * bmm(a, input) -> (b, m)
            
        #     w_grad[i,:] = sum(res / l) / b
        w_grad = torch.zeros_like(self.w)
        input_, output = self.saved_tensors
        
        grad = output_grad.output_grad
        b = grad.size(0)
        n = grad.size(1)
        
        a = torch.exp(self.beta * output)
        for i in range(n):
            a_i = a[:, :, i].unsqueeze(1)
            l = torch.sum(a_i, dim=-1).squeeze()
            res = grad[:, i].unsqueeze(-1) * torch.bmm(a_i, input_)
            
            w_grad[:, i] = torch.sum(res / l.unsqueeze(-1).unsqueeze(-1), dim=0) / b
        
        # Compute the gradient of the input
        grad_input = output_grad.output_grad @ self.w.t()  # Backpropagate through weights
        
        total_grad = ConnectionGradWrapper(grad_input, w_grad)
        self.update(total_grad)  # Update weights and bias

        return total_grad
    