import torch
from typing import Tuple
from omegaconf import DictConfig

from settings.serializable import YAMLSerializable
from network.learning.learning_rule import LearningRule


@YAMLSerializable.register("SingleSpikeLR")
class SingleSpikeLR(LearningRule, YAMLSerializable):
    """
    Single Spike Learning Rule
    """
    
    def __init__(self, threshold: float = 1.0):
        super().__init__()
        super(SingleSpikeLR, self).__init__()
        self._threshold = threshold
        self.saved_tensors = None
    
    def forward(self, input_):
        
        # Find the maximum value and its index along the appropriate dimension (supports batch processing).
        max_val, max_idx = torch.max(input_, dim=-input_.dim() + 1)
        
        # Save the index of the max value for use in the backward pass.
        self.saved_tensors = max_idx, max_val
        
        # Compute the difference between the max value and the threshold.
        threshold_diff = max_val - self._threshold
        
        return threshold_diff # TODO: make it return a spike data
    
    
    def backward(self, input_, E: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backward function for the layer. Computes the gradient of the output with respect to the input.
        This function uses the saved max index from the forward pass to help compute the gradient.

        Parameters:
        -----------
        E : torch.Tensor
            The gradient of the loss with respect to the output of the layer.
        """
        
        spike_indices, max_vals = self.saved_tensors
        
        # Check if input is a single sample or a batch
        if input_.dim() == 1:  # Single sample
            input_ = input_.unsqueeze(0)  # Add a batch dimension if necessary

        res = []
        
        # enumerating over batch data
        for i, idx in enumerate(spike_indices):
            res.append((E[i] @ input_[i, idx, :]).t()) #TODO: check if we can use max_vals[i] instead of input_[i, idx, :]
            
        weight_grad = torch.stack(res)

        return weight_grad
    
    @classmethod
    def from_config(cls, config: DictConfig, env_config: DictConfig):
        return cls(env_config.v_th)
