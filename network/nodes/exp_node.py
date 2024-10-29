import torch
import torch.autograd
# from typing import override

from network.nodes.node import Node
from common import Configuration, SPIKE_NS, MODEL_NS
from network.learning.grad_wrapper import GradWrapper

# PLASTICITY_INDUCTION = 1.0e-3
class ExpNode(Node):
    
    def __init__(
        self,
        config : Configuration,
        n,
        device=None,
        dtype=None,
        learning=False
    ):
        super(ExpNode, self).__init__(n, (n, n), learning)
        
        self._config = config
        self.device = device
        self._threshold = self._config[SPIKE_NS.v_thr]  # Threshold for firing spikes
        self._dt = self._config[SPIKE_NS.dt]  # Time step for spike calculation
        self._beta = self._config[MODEL_NS.BETA]
        self.saved_tensors = None  # Placeholder for saving intermediate values (used for backward pass)

    def forward(self, input_):
        """
        Forward function for the layer. This determines whether the neurons fire spikes by comparing the
        input to the threshold. If any input exceeds the threshold, it calculates the difference and returns
        the spike time and the difference from the threshold.

        Parameters:
        -----------
        input_ : torch.Tensor
            The input tensor representing incoming signals to the node.

        Returns:
        --------
        torch.Tensor:
            The difference between the maximum value of the input and the threshold for firing.
        """
        # Move input and target to the specified device (GPU or CPU)
        input_ = input_.to(self.device)

        # Ensure input and target are at least 2D (batch-wise)
        if input_.dim() == 1:
            input_ = input_.unsqueeze(0)  # Add batch dimension
        if input_.dim() == 3:
            input_ = input_.squeeze(1)
                
        exp = torch.exp(self._beta * input_)
        res = torch.sum(exp, dim=-2) * self._dt
        
        if torch.is_grad_enabled():
            self.saved_tensors = exp
        
        return torch.log(res) - self._threshold

    
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
        exp = self.saved_tensors
        
        n = output_grad.size(-1)
        
        for i in range(n):
            exp[:, :, i] = exp[:, :, i] / torch.sum(exp[:, :, i], dim=-1, keepdim=True)
        
        info = {"exp": exp}
        return GradWrapper(output_grad=output_grad, info=info)
