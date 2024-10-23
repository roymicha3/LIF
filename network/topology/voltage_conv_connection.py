import numpy as np
import torch


from network.nodes.node import Node
from network.learning.grad_wrapper import GradWrapper, ConnectionGradWrapper
from network.topology.simple_connection import SimpleConnection

class SimpleConnection(SimpleConnection):
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
        super().__init__(None, source, target, norm, device=device)
        self.beta = beta
        self.saved_tensors = None


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

    