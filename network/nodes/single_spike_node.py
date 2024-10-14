import torch
import torch.autograd
# from typing import override

from network.nodes.node import Node
from common import ATTR, SPIKE_NS
from network.learning.grad_wrapper import GradWrapper

class SingleSpikeNode(Node):
    """
    A class representing a node that produces a single spike.
    This node fires when the input signal exceeds a threshold, and it supports
    both learning and non-learning behaviors. The node returns spike times and
    the difference between the maximum input value and the threshold.

    Attributes:
    -----------
    n : int
        The number of neurons in the node.
    device : torch.device, optional
        The device on which the tensor operations will be performed (CPU or GPU).
    dtype : torch.dtype, optional
        The data type for the node's tensors.
    learning : bool, optional
        A flag to indicate whether learning is enabled for this node.
    _threshold : torch.Tensor
        The firing threshold for the neurons in the node.
    _dt : torch.Tensor
        The time step for spike time calculation.
    saved_tensors : torch.Tensor or None
        A tensor used to store intermediate values (e.g., max index) during forward pass for backward computation.

    Methods:
    --------
    __init__(self, n, device=None, dtype=None, learning=False):
        Initializes the SingleSpikeNode with the specified number of neurons and configurations.

    forward(self, input_):
        Computes the forward pass of the node, determining whether the input exceeds the threshold and
        returns the spike time (or the difference between max value and threshold).

    backward(self, output_grad):
        Computes the gradient of the loss with respect to the input, using the saved max index from the forward pass.
    """
    
    def __init__(
        self,
        n,
        device=None,
        dtype=None,
        learning=False
    ):
        """
        Constructor for SingleSpikeNode.

        Parameters:
        -----------
        n : int
            The number of neurons in this node.
        device : torch.device, optional
            The device on which to perform computations (default: None, meaning CPU is used).
        dtype : torch.dtype, optional
            The data type for the node's tensors (default: None).
        learning : bool, optional
            A flag indicating if the node should support learning (default: False).
        """
        super(SingleSpikeNode, self).__init__(n, (n, n), learning)
        self._threshold = ATTR(SPIKE_NS.v_thr)  # Threshold for firing spikes
        self._dt = ATTR(SPIKE_NS.dt)  # Time step for spike calculation
        self.saved_tensors = None  # Placeholder for saving intermediate values (used for backward pass)

    def forward(self, input_):
        """
        Forward function for the layer. This determines whether the neurons fire spikes by comparing the
        input to the threshold. If any input exceeds the threshold, it calculates the difference and returns
        the spike time and the difference from the threshold.

        Parameters:
        -----------
        input_ : torch.Tensor
            The input tensor representing incoming signals to the node.

        Returns:
        --------
        torch.Tensor:
            The difference between the maximum value of the input and the threshold for firing.
        """
        # Find the maximum value and its index along the appropriate dimension (supports batch processing).
        max_val, max_idx = torch.max(input_, dim=-input_.dim() + 1)
        
        # Save the index of the max value for use in the backward pass.
        self.saved_tensors = max_idx
        
        # Compute the difference between the max value and the threshold.
        threshold_diff = max_val - self._threshold
        
        return threshold_diff

    def backward(self, output_grad) -> GradWrapper:
        """
        Backward function for the layer. Computes the gradient of the output with respect to the input.
        This function uses the saved max index from the forward pass to help compute the gradient.

        Parameters:
        -----------
        output_grad : torch.Tensor
            The gradient of the loss with respect to the output of the layer.

        Returns:
        --------
        tuple of torch.Tensor:
            Gradients of the loss with respect to the input and the saved max index.
        """
        # Retrieve the max index saved during the forward pass.
        max_idx = self.saved_tensors
        
        # Return the gradient of the input and the saved index (used for further backprop).
        return GradWrapper(output_grad=output_grad, info={"max_idx": max_idx})