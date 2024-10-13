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
        self.saved_tensors = (input_, target_)
        max_val = input_[SingleSpikeNode.SPIKE_VAL_IDX]
        
        return self.relu( - torch.sum(target_ * max_val))
    
    def backward(self):
        """
        :param grad_output: Gradient of the loss with respect to the output of the layer.
        :return: Gradients with respect to the input and target tensors.
        """
        # Retrieve saved tensors from the forward pass
        input_, target_ = self.saved_tensors
        
        # Access the spike values
        spike_values = input_[SingleSpikeNode.SPIKE_VAL_IDX]
        
        # Compute the gradient of the loss with respect to spike values
        grad_spike_values = torch.zeros_like(spike_values)
        
        grad_spike_values[spike_values * target_ > 0] = 0
        grad_spike_values[spike_values * target_ <= 0] = 1

        # Grad with respect to input_
        grad_input = - target_ * grad_spike_values
        
        # Grad with respect to target_: dL/d_target_
        grad_target = - spike_values * grad_spike_values
        
        return grad_input, grad_target
        