from torch import nn
import torch

from network.nodes.single_spike_node import SingleSpikeNode

class BinaryLoss(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.relu = nn.ReLU()
        self.saved_tensors = None
    
    def forward(self, input_, target_) -> torch.Tensor:
        """
        :param input_: (spike times, spike values).
        :param target_: labels (either +1 or -1)
        """
        # Ensure input and target are at least 2D (batch-wise)
        if input_.dim() == 1:
            input_ = input_.unsqueeze(0)  # Add batch dimension
        
        if target_.dim() == 1:
            target_ = target_.unsqueeze(0)  # Add batch dimension
        
        self.saved_tensors = (input_, target_)
        
        return self.relu( - target_ * input_)
    
    def backward(self, grad_output=None):
        """
        :param grad_output: Gradient of the loss with respect to the output of the layer.
        :return: Gradients with respect to the input and target tensors.
        """
        # Retrieve saved tensors from the forward pass
        input_, target_ = self.saved_tensors
        
        # Compute the gradient of the loss with respect to spike values
        grad_spike_values = torch.zeros_like(input_)
        
        grad_spike_values[(input_ * target_) > 0] = 0
        grad_spike_values[(input_ * target_) <= 0] = 1

        # Grad with respect to input_
        grad_input = - target_ * grad_spike_values
        
        # Grad with respect to target_: dL/d_target_
        # Squeeze input_ to match target_ dimensions before multiplication
        grad_target = - input_.squeeze(dim=-1) * grad_spike_values.squeeze(dim=-1)

        return grad_input, grad_target
        