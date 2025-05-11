import torch
import matplotlib.pyplot as plt

from experiment_manager.environment import Environment

from tools.utils import SEQ_LEN
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
                 activation: Activation,
                 env: Environment) -> None:
        
        super().__init__()
        
        self.kernel = kernel
        self.connection = connection
        self.activation = activation
        self.env = env
        
        self.dt = env.args.dt
        self.T = env.args.T
        
        self.time_seq = SEQ_LEN(self.T, self.dt)
        
    
    def forward(self, input_):
        
        input_voltage = self.kernel.forward(input_)
        inner_voltage, spikes = self.connection.forward(input_voltage)
        output = self.activation.forward(inner_voltage)
        return output
    
    def __call__(self, input_):

        kernel = self.kernel(input_)
        
        res = []
        for t in range(self.time_seq):
            input_voltage_t = next(kernel)
            inner_voltage_t, spikes_t = self.connection(input_voltage_t)
            output_t = self.activation.forward(inner_voltage_t)
            
            res.append(output_t)
        
        output = torch.stack(res, dim=-1)
        if len(output.size()) == 2:
            output = output.unsqueeze(-1)
        
        # output = output.permute(1, 2, 0)
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
