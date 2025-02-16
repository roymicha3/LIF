import torch.nn.functional as F

from network.activation.activation import Activation
from settings.serializable import YAMLSerializable

@YAMLSerializable.register("SoftmaxActivation")
class SoftmaxActivation(Activation, YAMLSerializable):
    def __init__(self):
        super().__init__()
        super(YAMLSerializable).__init__()
        self.saved_vectors = None

    def forward(self, input_):
        self.saved_vectors = input_
        return F.softmax(input_, dim=-1)

    def backward(self, grad_output):
        if grad_output.dim() == 3:
            grad_output = grad_output.squeeze(-1)
        
        input_ = self.saved_vectors 
        softmax_output = F.softmax(input_, dim=-1)
        grad = grad_output * softmax_output * (1 - softmax_output)
        return grad.unsqueeze(-1)
    
    @classmethod
    def from_config(cls, config, env_config):
        return cls()