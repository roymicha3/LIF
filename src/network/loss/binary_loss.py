import torch
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.environment import Environment
from omegaconf import DictConfig
from torch import nn


@YAMLSerializable.register("BinaryLoss")
class BinaryLoss(nn.Module):
    
    def __init__(self, device=None, threshold = 0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.relu = nn.ReLU()
        self.saved_tensors = None
        self.device = device
        self.threshold = threshold
    
    def forward(self, input_, target_) -> torch.Tensor:
        """
        :param input_: (spike times, spike values).
        :param target_: labels (either +1 or -1)
        """
        # Move input and target to the specified device (GPU or CPU)
        input_ = input_.to(self.device)
        target_ = target_.to(self.device)

        # Ensure input and target are at least 2D (batch-wise)
        if input_.dim() == 1:
            input_ = input_.unsqueeze(0)  # Add batch dimension
        if input_.dim() == 3:
            input_ = input_.squeeze(1)
        
        if target_.dim() == 1:
            target_ = target_.unsqueeze(0)  # Add batch dimension
        
        outputs = input_ - self.threshold
        self.saved_tensors = (outputs, target_)
        
        # Compute and return the loss, applying ReLU after multiplying target and input
        return self.relu(- target_ * outputs)
    
    def backward(self, grad_output=None):
        """
        :param grad_output: Gradient of the loss with respect to the output of the layer.
        :return: Gradients with respect to the input and target tensors.
        """
        # Retrieve saved tensors from the forward pass
        outputs, target_ = self.saved_tensors
        
        # Compute the gradient of the loss with respect to spike values
        grad_spike_values = torch.zeros_like(outputs).to(self.device)
        
        epsilon = 0  # Small value
        
        grad_spike_values[(outputs * target_) > -epsilon] = 0
        grad_spike_values[(outputs * target_) <= -epsilon] = 1

        # Grad with respect to outputs
        grad_input = - target_ * grad_spike_values
        
        return grad_input.unsqueeze(-1)
    
    def classify(self, data: torch.Tensor) -> torch.Tensor:
        # Move the data to the correct device before classifying
        data = data.to(self.device)
        
        # Classify based on whether values are greater than 0 or not
        predicted = (data.squeeze() > 0) * 2 - 1
        return predicted
    
    @staticmethod
    def from_config(config: DictConfig, env: Environment) -> "BinaryLoss":
        threshold = config.get("threshold", 0.0)
        return BinaryLoss(env.device, threshold)
