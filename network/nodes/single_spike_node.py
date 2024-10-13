import torch
import torch.autograd
# from typing import override

from network.nodes.node import Node
from common import ATTR, SPIKE_NS

class MaxTimeGrad(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, input_, dt, threshold):
        """
        Forward pass of the max time training method.
        Returns a tensor with two values: (max_val - threshold, max_time).
        """
        # Check if any input exceeds the threshold
        max_val, max_idx = torch.max(input_, dim=-input_.dim() + 1)
        max_time = max_idx * dt
        
        # Compute the difference from the threshold
        threshold_diff = max_val - threshold
        
        # Stack the results into a single tensor
        return torch.stack((threshold_diff, max_time), dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass of the max time training method - this layer has no parameters to train
        """
        # Since dt and threshold are not trainable parameters, we return None for them
        return grad_output, None, None

# Define the single spike Node class (a neuron that fires once if the max output voltage reaches a certain threshold)
class SingleSpikeNode(Node):
    
    NO_SPIKE = 0
    
    SPIKE_TIME_IDX = 0
    SPIKE_VAL_IDX = 1
    SPIKE_IDX = 2
    
    def __init__(
        self,
        n,
        device=None,
        dtype=None,
        learning = False,
        grad_function : torch.autograd.Function = MaxTimeGrad
    ):
        super(SingleSpikeNode, self).__init__(n, (n, n), learning)
        self._threshold = ATTR(SPIKE_NS.v_thr)
        self._dt = ATTR(SPIKE_NS.dt)
        self._grad_function = grad_function
        
    # @override
    def forward(self, input_):
        """
        Forward function for the layer, returns the spike time or NO_SPIKE value.
        Supports multiple output neurons by handling dimensions properly.
        Returns: (max value time, max value - the threshold)
        """
        return self._grad_function.apply(input_, self._dt, self._threshold)
    
    
