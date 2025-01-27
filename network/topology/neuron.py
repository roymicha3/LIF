import torch

from common.constants import SPIKE_NS
from network.topology.kernel import Kernel
from network.topology.connection import Connection
from network.learning.learning_rule import LearningRule

class NeuronLayer(torch.nn.Module):
    """
    Base class for neurons.
    """
    
    def __init__(self,
                 config,
                 kernel: Kernel,
                 connection: Connection,
                 learning_rule: LearningRule) -> None:
        
        super().__init__()
        self._config = config
        self._dt = self._config[SPIKE_NS.dt]  # Time step for spike calculation
        
        self.kernel = kernel
        self.connection = connection
        self.learning_rule = learning_rule
        
    
    def forward(self, input_):
        
        # Find the maximum value and its index along the appropriate dimension (supports batch processing).
        max_val, max_idx = torch.max(input_, dim=-input_.dim() + 1)
        
        # Save the index of the max value for use in the backward pass.
        self.saved_tensors = max_idx, max_val
        
        # Compute the difference between the max value and the threshold.
        threshold_diff = max_val - self._threshold
        
        return threshold_diff

    def backward(self, output_grad) -> GradWrapper:
        """
        Backward function for the layer. Computes the gradient of the output with respect to the input.
        This function uses the saved max index from the forward pass to help compute the gradient.

        Parameters:
        -----------
        output_grad : torch.Tensor
            The gradient of the loss with respect to the output of the layer.

        Returns:
        --------
        tuple of torch.Tensor:
            Gradients of the loss with respect to the input and the saved max index.
        """
        # Retrieve the max index saved during the forward pass.
        max_idx, max_val = self.saved_tensors
        
        info = {
            "max_idx": max_idx,
            # "plasticity_induction": plasticity_induction
            }
        return GradWrapper(output_grad=output_grad, info=info)
