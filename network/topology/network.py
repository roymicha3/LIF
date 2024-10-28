import torch
import tempfile
from typing import Tuple

from common import Configuration
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
        config: Configuration,
        learning: bool = True,
        device = None
    ) -> None:
        """
        Initializes network object.

        :param dt: Simulation timestep
        :param learning: Whether to allow connection updates. True by default.
        """
        super().__init__()
        
        self.config = config
        self.learning = learning

        self.layers = {}
        self.connections = {}
        self.monitors = {}
        
        self.device = device

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
    
    def _get_prev_connection(self, next_connection: Tuple[str, str] = None) -> Tuple[str, str]:            
        
        if Network.INPUT_LAYER_NAME not in self.layers.keys() or Network.OUTPUT_LAYER_NAME not in self.layers.keys():
            raise Exception("missing Input and/or Output layer")
        
        target = Network.OUTPUT_LAYER_NAME
        
        if next_connection is not None:
            target = next_connection[0]
        
        for (source_name, target_name), connection in self.connections.items():
            if target_name == target:
                return (source_name, target_name)
            
        return None

    
    def forward(self, data: torch.Tensor) -> None:
        """
        forward function of the network
        """
        current_connection = self._get_next_connection()
        
        while current_connection:
            source, _ = current_connection
            data = self.layers[source].forward(data)
            data = self.connections[current_connection].forward(data)
            
            current_connection = self._get_next_connection(current_connection)

        data = self.layers[Network.OUTPUT_LAYER_NAME].forward(data)
        return data
    
    def backward(self, grad: torch.Tensor) -> None:
        """
        the backward function of the network
        """
        
        current_connection = self._get_prev_connection()
        
        while current_connection:
            _, target = current_connection
            grad = self.layers[target].backward(grad)
            grad = self.connections[current_connection].backward(grad)
            
            self.connections[current_connection].update(grad)
            
            current_connection = self._get_prev_connection(current_connection)
            

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