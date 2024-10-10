import torch
# from typing import override

from .node import Node
from .leaky_node import LeakyNode
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
        forward function for the layer, returns the spike time or NO_SPIKE value
        """
        if torch.max(input_) >= self._threshold:
            return torch.argmax(input_, dim=None, keepdim=False).long() * self._dt
        
        return torch.tensor(SingleSpikeNode.NO_SPIKE)
    
