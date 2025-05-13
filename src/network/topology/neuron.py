import torch
import matplotlib.pyplot as plt
from network.kernel.kernel import Kernel
from network.topology.connection import Connection
from network.activation.activation import Activation

class NeuronLayer(torch.nn.Module):
    """
    Base class for neurons.
    """
    
    def __init__(self,
                 kernel: Kernel,
                 connection: Connection,
                 activation: Activation) -> None:
        
        super().__init__()
        
        self.kernel = kernel
        self.connection = connection
        self.activation = activation
        
    
    def forward(self, input_):
        input_voltage = self.kernel.forward(input_)
        inner_voltage = self.connection.forward(input_voltage)
        output = self.activation.forward(inner_voltage)
        return output
        
    def backward(self, output_grad):
        """
        Backward function for the neuron layer.
        """
        activation_grad = self.activation.backward(output_grad)
        input_grad = self.connection.backward(activation_grad)
        return self.kernel.backward(input_grad)
    
    def partial_forward(self, input_):
        input_voltage = self.kernel.forward(input_)
        inner_voltage = self.connection.partial_forward(input_voltage)
        return inner_voltage
    
    def plot(self, input_):
        inner_voltage = self.partial_forward(input_)
        
        # plot the inner voltage
        plt.plot(inner_voltage)
        plt.title("Inner Voltage")
        plt.xlabel("Time")
        plt.ylabel("Voltage")
        plt.show()
