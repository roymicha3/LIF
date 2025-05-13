import torch
from typing import Tuple
from omegaconf import DictConfig

from network.learning.learning_rule import LearningRule

from experiment_manager.environment import Environment
from experiment_manager.common.serializable import YAMLSerializable


@YAMLSerializable.register("SequentialSingleSpikeLR")
class SequentialSingleSpikeLR(LearningRule, YAMLSerializable):
    """
    Sequential Single Spike Learning Rule
    """
    
    def __init__(self, threshold: float = 1.0):
        super().__init__()
        super(SequentialSingleSpikeLR, self).__init__()
        self._threshold = threshold
        self.saved_tensors = None
        self.max_values = None
        self.t_max = None
        self.t = 0
    
    def forward(self, input_, output_, **kwargs) -> torch.Tensor:
        
        if self.max_values is None:
            self.max_values = input_.clone()
            self.saved_tensors = input_.clone()
            self.t_max = torch.zeros_like(output_, dtype=torch.long)
        
        
        self.max_values = torch.max(self.max_values, output_)
        indices = (self.max_values == output_)
        self.t_max[indices] = self.t
            
        self.saved_tensors[indices] = input_[indices]
            
        self.t += 1
        return self.max_values - self._threshold # TODO: make it return a spike data
    
    
    def backward(self, input_, E: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backward function for the layer. Computes the gradient of the output with respect to the input.
        This function uses the saved max index from the forward pass to help compute the gradient.

        Parameters:
        -----------
        E : torch.Tensor
            The gradient of the loss with respect to the output of the layer.
        """
        
        # Check if input is a single sample or a batch
        if input_.dim() == 1:  # Single sample
            input_ = input_.unsqueeze(0)  # Add a batch dimension if necessary

        weight_grad = E @ self.saved_tensors
        
        # reset the saved tensors
        self.saved_tensors = None
        self.max_values = None
        self.t_max = None
        self.t = 0
        
        return weight_grad
    
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment):
        return cls(env.args.v_th)
