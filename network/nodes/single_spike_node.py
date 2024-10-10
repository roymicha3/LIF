import torch
# from typing import override

from network.nodes.node import Node
from network.nodes.leaky_node import LeakyNode
from common import ATTR, SPIKE_NS
from data.data_sample import DataSample

# Define the single spike Node class (a neuron that fires once if the max output voltage reaches a certain threshold)
class SingleSpikeNode(Node):
    
    NO_SPIKE = 0
    
    def __init__(
        self,
        n,
        device=None,
        dtype=None,
        learning = False
    ):
        super(SingleSpikeNode, self).__init__(n, (n, n), learning)
        self._threshold = ATTR(SPIKE_NS.v_thr)
        self._dt = ATTR(SPIKE_NS.dt)
        
    # @override
    def forward(self, input_):
        """
        Forward function for the layer, returns the spike time or NO_SPIKE value.
        Supports multiple output neurons by handling dimensions properly.
        """
        # Check if any input exceeds the threshold
        max_val, max_idx = torch.max(input_, dim=-1)
        
        # If any of the inputs exceed the threshold, return the spike time
        if (max_val >= self._threshold).any():
            return max_idx.long() * self._dt

        # Otherwise, return NO_SPIKE value
        return torch.tensor(SingleSpikeNode.NO_SPIKE, device=input_.device)
    
