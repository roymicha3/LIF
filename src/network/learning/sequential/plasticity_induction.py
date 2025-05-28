from typing import Tuple

import torch
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.environment import Environment
from omegaconf import DictConfig

from network.learning.learning_rule import LearningRule


@YAMLSerializable.register("PlasticityInduction")
class PlasticityInduction(LearningRule, YAMLSerializable):
    
    def __init__(self, threshold: float = 1.0, epsilon: float = 0.01):
        super().__init__()
        super(PlasticityInduction, self).__init__()
        self._threshold = threshold
        self._epsilon = epsilon
        self.saved_tensors = None
    
    def forward(self, input_, output_, **kwargs) -> torch.Tensor:
        
        if self.saved_tensors is None:
            self.saved_tensors = torch.ones(
                size=(input_.size(-1), output_.size(-1)),
                dtype=torch.float32,
                device=input_.device)
            
            
        indices = (self._threshold < output_) # neurons that fired
        
        for b in range(input_.size(0)):
            self.saved_tensors[b, :, indices[b]] = 0 # overall the nerons that didnt fire
        
        return 0
    
    
    def backward(self, input_, E: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Check if input is a single sample or a batch
        if input_.dim() == 1:  # Single sample
            input_ = input_.unsqueeze(0)  # Add a batch dimension if necessary
        
        weight_grad = torch.zeros_like(self.saved_tensors, device=input_.device)
        
        for b in range(input_.size(0)):
            # Compute the gradient of the loss with respect to the weights
            silent_output_neurons = E[b] < 0
            weight_grad[b, :, silent_output_neurons] = \
                self.saved_tensors[b, :, silent_output_neurons] * self._epsilon
        
        # reset the saved tensors
        self.reset()
        
        return weight_grad
    
    def reset(self):
        """
        Reset the saved tensors.
        """
        self.saved_tensors = None
    
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment):
        return cls(env.args.v_th, config.epsilon)
