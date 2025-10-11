import torch
from typing import List

from experiment_manager.environment import Environment
from experiment_manager.common.serializable import YAMLSerializable

from tools.utils import SEQ_LEN
from network.topology.network import Network
from data.spike.spike_sample import SpikeSample, digest_batch


@YAMLSerializable.register("SequentialNetwork")
class SequentialNetwork(Network):
    
    def __init__(self, config: dict, env: Environment, learning: bool = True, device=None):
        super().__init__(config, env, learning=learning, device=device)
        self.env = env
        self.dt = env.args.dt
        self.T = env.args.T
        
        self.time_seq = SEQ_LEN(self.T, self.dt)

    def add_layer(self, layer, name):
        """
        Add a layer to the network.
        """
        self.layers.append(layer)
        self.add_module(name, layer)

        layer.train(self.learning)
        
    def forward(self, data: List[SpikeSample]) -> None:
        """
        forward function of the network
        """
        # TODO: this function recieves spike samples and so is the neuron __call__ function,
        # but the second iteration returns voltage tensor
        for layer in self.layers:
            data, spikes = layer(data, spikes=False)
            
        return data, spikes
    
    def inner_state(self, input_, layer_idx: int):
        """
        return the inner state of a layer of a given index
        """
        inner_layer = self.layers[layer_idx]
        data = input_
        for layer in self.layers:
            data = layer(data)
            
            if layer == inner_layer:
                break
            
        return data
    
    
    def backward(self, grad: torch.Tensor) -> None:
        """
        the backward function of the network
        """
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad)