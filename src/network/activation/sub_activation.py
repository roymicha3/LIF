from network.activation.activation import Activation

from experiment_manager.environment import Environment
from experiment_manager.common.serializable import YAMLSerializable

@YAMLSerializable.register("SubtractActivation")
class SubtractActivation(Activation, YAMLSerializable):
    def __init__(self, threshold: float):
        super().__init__()
        super(YAMLSerializable).__init__()
        self.threshold = threshold

    def forward(self, input):
        return input - self.threshold

    def backward(self, grad_output):
        # Assuming the gradient is simply passed through
        return grad_output
    
    @classmethod
    def from_config(cls, config, env: Environment) -> "SubtractActivation":
        return cls(env.args.v_th)