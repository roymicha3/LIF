import torch

class GradWrapper:
    
    def __init__(self, grad: torch.Tensor, info: dict = {}):
        self._grad = grad
        self._info = info
        
    def get(self) -> torch.Tensor:
        return self._grad
    
    def set_grad(self, new_grad: torch.Tensor):
        self._grad = new_grad
        
    def set_info(self, new_info: dict):
        self._info = new_info
    
    def additional_info(self, key) -> any:
        return self._info[key]
    

class ConnectionGradWrapper(GradWrapper):
    
    def __init__(self, output_grad: torch.Tensor, weight_grad: torch.Tensor, info = {}):
        super().__init__(output_grad, info)
        self._weight_grad = weight_grad
        
    def get_weight_grad(self):
        return self._weight_grad
