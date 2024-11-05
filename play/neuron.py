import torch
import torch.nn as nn

from network.learning.grad_wrapper import GradWrapper

class Neuron(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self._config = config
        
        
    def forward(self, input_spikes):
        raise NotImplementedError
    
    def backward(self, output_grad: GradWrapper):
        raise NotImplementedError