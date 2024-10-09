from torch import nn
import torch

class BinaryLoss(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass
        