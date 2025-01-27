import torch

from common import SPIKE_NS
from network.kernel.kernel import Kernel
from network.topology.connection import Connection
from network.learning.learning_rule import LearningRule

class NeuronLayer(torch.nn.Module):
    """
    Base class for neurons.
    """
    
    def __init__(self,
                 config,
                 kernel: Kernel,
                 connection: Connection) -> None:
        
        super().__init__()
        self._config = config
        self._dt = self._config[SPIKE_NS.dt]  # Time step for spike calculation
        
        self.kernel = kernel
        self.connection = connection
        
    
    def forward(self, input_):
        input_voltage = self.kernel.forward(input_)
        output = self.connection.forward(input_voltage)
        return output
        
    def backward(self, output_grad):
        """
        Backward function for the neuron layer.
        """
        input_grad = self.connection.backward(output_grad)
        return self.kernel.backward(input_grad)