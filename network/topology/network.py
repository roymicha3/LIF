import torch
import tempfile
from typing import Tuple

from network.topology.neuron import NeuronLayer


class Network(torch.nn.Module):
    """
    Responsible for the simulation and interaction of nodes and connections.
    """

    def __init__(
        self,
        config: dict,
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

        self.layers = []
        
        self.device = device

    def add_layer(self, layer: NeuronLayer, name: str) -> None:
        """
        Adds a layer of nodes to the network.

        :param layer: A subclass of the ``Nodes`` object.
        :param name: Logical name of layer -> the network must have an ''Input'' layer.
        """
        self.layers.append(layer)
        self.add_module(name, layer)

        layer.train(self.learning)


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

    
    def forward(self, data: torch.Tensor) -> None:
        """
        forward function of the network
        """
        
        for layer in self.layers:
            data = layer.forward(data)
            
        return data
    
    def backward(self, grad: torch.Tensor) -> None:
        """
        the backward function of the network
        """
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
            

    def reset_state_variables(self) -> None:
        """
        Reset state variables of objects in network.
        """
        for layer in self.layers:
            self.layers[layer].reset_state_variables()

    def train(self, mode: bool = True) -> "torch.nn.Module":
        """
        Sets the node in training mode.

        :param mode: Turn training on or off.

        :return: ``self`` as specified in ``torch.nn.Module``.
        """
        self.learning = mode
        return super().train(mode)

    def parameters(self, recurse: bool = True):
        """
        Returns an iterator over module parameters that are trainable.
        """
        for name, param in super().named_parameters(recurse=recurse):
            if param.requires_grad:
                yield param

    def named_parameters(self, prefix: str = '', recurse: bool = True):
        """
        Returns an iterator over module named parameters that are trainable.
        """
        for name, param in super().named_parameters(prefix=prefix, recurse=recurse):
            if param.requires_grad:
                yield name, param
