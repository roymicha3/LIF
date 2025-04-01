import torch
from omegaconf import DictConfig

import torch.nn as nn
from settings.serializable import YAMLSerializable 


@YAMLSerializable.register("CrossEntropyLoss")
class CrossEntropyLoss(nn.Module):
    
    def __init__(self, device=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device
    
    def forward(self, input_, target_) -> torch.Tensor:
        """
        :param input_: logits (raw, unnormalized scores for each class).
        :param target_: labels (class indices).
        """
        # Move input and target to the specified device (GPU or CPU)
        input_ = input_.to(self.device)
        target_ = target_.to(self.device)

        # Ensure input and target are at least 2D (batch-wise)
        if input_.dim() == 1:
            input_ = input_.unsqueeze(0)  # Add batch dimension
        
        if target_.dim() == 1:
            target_ = target_.unsqueeze(0)  # Add batch dimension
        
        # Compute and return the cross entropy loss
        return self.loss_fn(input_, target_.squeeze())
    
    def backward(self, input_, target_):
        input_.require_grad = True
        input_ = input_.to(self.device)
        target_ = target_.to(self.device)
        
        loss = self.forward(input_, target_)
        grad = torch.autograd.grad(loss, input_, create_graph=True)[0]
        return grad
    
    def classify(self, data: torch.Tensor) -> torch.Tensor:
        data = data.to(self.device)
        
        # Classify based on the highest logit value
        predicted = torch.argmax(data, dim=-1)
        return predicted
    
    @staticmethod
    def from_config(config: DictConfig, env_config: DictConfig):
        return CrossEntropyLoss(env_config.device)
