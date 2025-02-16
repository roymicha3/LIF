from settings.factory import Factory
from settings.serializable import YAMLSerializable

# import all activations
from network.activation.sub_activation import SubtractActivation
from network.activation.softmax import SoftmaxActivation

class ActivationFactory(Factory):
    
    @staticmethod
    def create(name, config, env_config):
        return YAMLSerializable.get_by_name(name).from_config(config, env_config)