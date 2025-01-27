import torch
from typing import Tuple

from network.learning.learning_rule import LearningRule

class SingleSpikeLR(LearningRule):
    """
    Single Spike Learning Rule
    """
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self._threshold = config["threshold"]
        self.saved_tensors = None
    
    def forward(self, input_):
        
        # Find the maximum value and its index along the appropriate dimension (supports batch processing).
        max_val, max_idx = torch.max(input_, dim=-input_.dim() + 1)
        
        # Save the index of the max value for use in the backward pass.
        self.saved_tensors = max_idx, max_val
        
        # Compute the difference between the max value and the threshold.
        threshold_diff = max_val - self._threshold
        
        return threshold_diff
    def backward(self, input_data, output_data, E: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backward function for the layer. Computes the gradient of the output with respect to the input.
        This function uses the saved max index from the forward pass to help compute the gradient.

        Parameters:
        -----------
        E : torch.Tensor
            The gradient of the loss with respect to the output of the layer.
        """
        # Retrieve the max index saved during the forward pass.
        max_idx, max_val = self.saved_tensors
        
        # TODO: implement me!
        
        raise NotImplementedError("Implement this function")