import tempfile
from typing import Dict, Iterable, Optional, Type, Tuple

import torch

from common import ATTR, SPIKE_NS
from network.learning.learning_rule import LearningRule
from network.nodes.node import Node
from network.topology.connection import Connection


class Network(torch.nn.Module):
    """
    Responsible for the simulation and interaction of nodes and connections.
    """

    INPUT_LAYER_NAME = "Input"
    OUTPUT_LAYER_NAME = "Output"

    def __init__(
        self,
        batch_size: int = 1,
        learning: bool = True
    ) -> None:
        """
        Initializes network object.

        :param dt: Simulation timestep.
        :param batch_size: Mini-batch size.
        :param learning: Whether to allow connection updates. True by default.
        """
        super().__init__()
        
        self.batch_size = batch_size
        self.learning = learning

        self.layers = {}
        self.connections = {}
        self.monitors = {}


    def add_layer(self, layer: Node, name: str) -> None:
        """
        Adds a layer of nodes to the network.

        :param layer: A subclass of the ``Nodes`` object.
        :param name: Logical name of layer -> the network must have an ''Input'' layer.
        """
        self.layers[name] = layer
        self.add_module(name, layer)

        layer.train(self.learning)

    def add_connection(
        self, connection: Connection, source: str, target: str
    ) -> None:
        """
        Adds a connection between layers of nodes to the network.

        :param connection: An instance of class ``Connection``.
        :param source: Logical name of the connection's source layer.
        :param target: Logical name of the connection's target layer.
        """
        self.connections[(source, target)] = connection
        self.add_module(source + "_to_" + target, connection)

        connection.train(self.learning)

    # TODO: see how to implement this monitor logic :)
    # def add_monitor(self, monitor: AbstractMonitor, name: str) -> None:
    #     """
    #     Adds a monitor on a network object to the network.

    #     :param monitor: An instance of class ``Monitor``.
    #     :param name: Logical name of monitor object.
    #     """
    #     self.monitors[name] = monitor
    #     monitor.network = self
    #     monitor.dt = self.dt

    def save(self, file_name: str) -> None:
        """
        Serializes the network object to disk.
        """
        torch.save(self, open(file_name, "wb"))

    def clone(self) -> "Network":
        """
        Returns a cloned network object.
        :return: A copy of this network.
        """
        virtual_file = tempfile.SpooledTemporaryFile()
        torch.save(self, virtual_file)
        virtual_file.seek(0)
        return torch.load(virtual_file)
    
    def _get_next_connection(self, prev_connection: Tuple[str, str] = None) -> Tuple[str, str]:            
        if Network.INPUT_LAYER_NAME not in self.layers.keys() or Network.OUTPUT_LAYER_NAME not in self.layers.keys():
            raise Exception("missing Input and/or Output layer")
        
        source = Network.INPUT_LAYER_NAME
        
        if prev_connection is not None:
            source = prev_connection[1]
        
        for (source_name, target_name), connection in self.connections.items():
            if source_name == source:
                return (source, target_name)
            
        return None

    def _get_output(self, data: torch.Tensor, layers: Iterable = None) -> Dict[str, torch.Tensor]:
        """
        Fetches outputs from network layers to use as input to downstream layers.

        :param layers: Layers to update inputs for. Defaults to all network layers.
        :return: Inputs to all layers for the current iteration.
        """

        if layers is None:
            layers = self.layers.keys()
            
        current_connection = self._get_next_connection()
        
        while current_connection:
            source, target = current_connection
            data = self.layers[source].forward(data)
            data = self.connections[current_connection].forward(data)
            data = self.layers[target].forward(data)
            
            current_connection = self._get_next_connection(current_connection)

        return data


    def run(
        self, data: Dict[str, torch.Tensor], **kwargs) -> None:
        """
        Simulate network for given inputs and time.

        :param inputs: Dictionary of ``Tensor``s of shape ``[time, *input_shape]`` or
                      ``[time, batch_size, *input_shape]``.
                      
        Keyword arguments:
        
        :param Bool progress_bar: Show a progress bar while running the network.

        **Example:**

        .. code-block:: python

            import torch
            import matplotlib.pyplot as plt

            from network import Network
            from network.nodes import Input
            from network.monitors import Monitor

            # Build simple network.
            network = Network()
            network.add_layer(Input(500), name='I')
            network.add_monitor(Monitor(network.layers['I'], state_vars=['s']), 'I')

            # Generate spikes by running Bernoulli trials on Uniform(0, 0.5) samples.
            spikes = torch.bernoulli(0.5 * torch.rand(500, 500))

            # Run network simulation.
            network.run(inputs={'I' : spikes})

            # Look at input spiking activity.
            spikes = network.monitors['I'].get('s')
            plt.matshow(spikes, cmap='binary')
            plt.xticks(()); plt.yticks(());
            plt.xlabel('Time'); plt.ylabel('Neuron index')
            plt.title('Input spiking')
            plt.show()
        """
        
        # Dynamic setting of batch size
        # goal shape is [time, batch, n_0, ...]
        for key, value in data.items():
            if value.dim() == 1:
                data[key] = value.unsqueeze(0).unsqueeze(0)
                
            elif value.dim() == 2:
                data[key] = value.unsqueeze(1)
                
            elif value.dim() == 3:
                pass
            else:
                raise Exception("Invalid input dimensions")
            
        outputs = {}
        
        # iterate over batches
        for key, value in data.items():
            outputs[key] = self._get_output(value)
        
        return outputs

    def reset_state_variables(self) -> None:
        """
        Reset state variables of objects in network.
        """
        for layer in self.layers:
            self.layers[layer].reset_state_variables()

        for connection in self.connections:
            self.connections[connection].reset_state_variables()

        # for monitor in self.monitors:
        #     self.monitors[monitor].reset_state_variables()

    def train(self, mode: bool = True) -> "torch.nn.Module":
        """
        Sets the node in training mode.

        :param mode: Turn training on or off.

        :return: ``self`` as specified in ``torch.nn.Module``.
        """
        self.learning = mode
        return super().train(mode)
