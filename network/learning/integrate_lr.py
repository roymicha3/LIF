import torch
from typing import Tuple
from omegaconf import DictConfig

from settings.serializable import YAMLSerializable
from network.learning.learning_rule import LearningRule


@YAMLSerializable.register("IntegrateLearningRule")
class IntegrateLearningRule(LearningRule, YAMLSerializable):
    """
    Integrate Learning Rule
    """
    
    def __init__(self, dt: float, threshold: float = 1.0, beta: float = 1.0):
        super().__init__()
        super(IntegrateLearningRule, self).__init__()
        self.dt = dt
        self.threshold = threshold
        self.beta = beta
        self.saved_tensors = None
    
    def forward(self, input_):
        
        exp = torch.exp(self.beta * input_)
        res = torch.sum(exp, dim=-2) * self.dt
        
        if torch.is_grad_enabled():
            self.saved_tensors = exp.detach()
        
        return torch.log(res) - self.threshold
    
    
    def backward(self, input_, E: torch.Tensor, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Backward function for the layer. Computes the gradient of the output with respect to the input.
        This function uses the saved max index from the forward pass to help compute the gradient.

        Parameters:
        -----------
        E : torch.Tensor
            The gradient of the loss with respect to the output of the layer.
        """
        
        exp = self.saved_tensors
        
        # Check if input is a single sample or a batch
        if input_.dim() == 1:  # Single sample
            input_ = input_.unsqueeze(0)  # Add a batch dimension if necessary

        b = E.size(0)
        
        h = E * self.beta * torch.bmm(exp.transpose(1, 2), input_)
        h = h / torch.sum(exp, dim=-2, keepdim=True)
        
        gradient = torch.sum(h, dim=0).t() / b # sum over all batches
        
        return gradient
    
    @classmethod
    def from_config(cls, config: DictConfig, env_config: DictConfig):
        return cls(
            env_config.dt,
            env_config.v_th,
            config.beta)
