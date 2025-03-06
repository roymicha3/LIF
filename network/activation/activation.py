from abc import ABC, abstractmethod

import torch.nn as nn
    
class Activation(nn.Module, ABC):

    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def backward(self, grad_output):
        pass
