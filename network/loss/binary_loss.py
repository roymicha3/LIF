from torch import nn
import torch

from network.nodes.single_spike_node import SingleSpikeNode

class BinaryLoss(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.relu = nn.ReLU()
        
        
    def forward(self, input_: torch.Tensor, target_: torch.Tensor) -> torch.Tensor:
        """
        :param input_: (spike times, spike values).
        :param target_: labels (either +1 or -1)
        """
        return self.relu( - target_ * input_[SingleSpikeNode.SPIKE_VAL_IDX])
        