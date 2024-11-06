import torch
from enum import Enum
import torch.nn as nn

from network.learning.grad_wrapper import GradWrapper

class NeuronOutputType(Enum):
    SPIKE = 1
    VALUE = 2

class Neuron(nn.Module):
    
    def __init__(self, config, type_: NeuronOutputType = NeuronOutputType.SPIKE):
        super().__init__()
        self._config = config
        self._type = type_
        
    def forward(self, input_spikes):
        raise NotImplementedError
    
    def backward(self, output_grad: GradWrapper):
        raise NotImplementedError