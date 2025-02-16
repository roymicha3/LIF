import torch.nn.functional as F

from network.activation.activation import Activation
from settings.serializable import YAMLSerializable

@YAMLSerializable.register("SoftmaxActivation")
class SoftmaxActivation(Activation, YAMLSerializable):
    def __init__(self):
        super().__init__()
        super(YAMLSerializable).__init__()

    def forward(self, input):
        return F.softmax(input, dim=-1)

    def backward(self, input, grad_output):
        softmax_output = F.softmax(input, dim=-1)
        return grad_output * softmax_output * (1 - softmax_output)
    
    @classmethod
    def from_config(cls, config, env_config):
        return cls()