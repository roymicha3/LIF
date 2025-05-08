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
        # TODO: remember to reset the state after each batch!
        generator = digest_batch(data)
        
        for t in range(self.time_seq):
            data_t = next(generator)
            res = []
            for layer in self.layers:
                data_t = layer.forward(data_t)
            
            res.append(data_t)
        
        res = torch.stack(res, dim=0)
        # TODO: check if the data is in the right shape
        # data = data.permute(1, 0, 2)
        
        return res
    
    def inner_state(self, input_, layer_idx: int):
        """
        return the inner state of a layer of a given index
        """
        inner_layer = self.layers[layer_idx]
        
        for t in range(self.time_seq):
            data_t = input_[t]
            
            for layer in self.layers:
                if layer is inner_layer:
                    return inner_layer.partial_forward(data_t)
                
                data_t = layer.forward(data_t)
        
        raise IndexError
    
    def backward(self, grad: torch.Tensor) -> None:
        """
        the backward function of the network
        """
        
        for layer in reversed(self.layers):
            grad = layer.backward(grad)