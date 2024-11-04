import torch
import torch.nn as nn

from network.learning.grad_wrapper import GradWrapper

class Neuron(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
    def forward(self, input_spikes):
        raise NotImplementedError
    
    def backward(self, output_grad: GradWrapper):
        raise NotImplementedError