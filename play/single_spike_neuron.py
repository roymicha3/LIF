from network.nodes.node import Node
from network.topology.connection import Connection
from play.neuron import Neuron
from data.spike.spike_sample import SpikeSample

class SingleSpikeNeuron(Neuron):
    
    def __init__(self, kernel: Node, connection: Connection, threshold: float = 1.0):
        super().__init__()
        self._kernel = kernel
        self._connection = connection
        self._threshold = threshold
        
    def forward(self, input_spikes):
        return super().forward(input_spikes)
    
    def backward(self, output_grad):
        return super().backward(output_grad)
    
    def _prepare(self, input_spikes: SpikeSample):
        spike_tensor = input_spikes.to_torch()
        input_voltage = self._kernel.forward(spike_tensor)
        v = self._connection.forward(input_voltage)
        k = self._connection.