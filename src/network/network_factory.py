from experiment_manager.common.factory import Factory
from experiment_manager.common.serializable import YAMLSerializable
from experiment_manager.environment import Environment
from omegaconf import DictConfig

from network.activation.activation_factory import ActivationFactory
from network.kernel.kernel_factory import KernelFactory
from network.learning.lr_factory import LearningRuleFactory
# import networks:
from network.topology.network import Network
from network.topology.neuron import NeuronLayer
from network.topology.sequential_network import SequentialNetwork
from network.topology.simple_connection import SimpleConnection


class NetworkFactory(Factory):
    """
    A factory class to build different types of neural network models with specified configurations.
    Currently supports:
    - A simple network with a direct connection between layers.
    - A network with a voltage-based convolutional connection.

    Methods:
    - build_simple_network: Constructs a network with a simple, direct connection.
    - build_voltage_convolution_network: Constructs a network with a voltage convolutional connection.
    """
    
    @staticmethod
    def create(name: str, config: DictConfig, env: Environment) -> Network:
        network_type = YAMLSerializable.get_by_name(name)
        network = network_type(config, env, learning=True, device=env.device)
        
        for layer in config.layers:
            kernel          = KernelFactory.create(layer.kernel.type, layer.kernel, env)
            
            lr_list = []
            if layer.get("learning_rule", None) is None:
                for lr in layer.learning_rules:
                    lr_list.append(LearningRuleFactory.create(lr.type, lr, env))
            else:
                lr_list.append(LearningRuleFactory.create(layer.learning_rule.type, layer.learning_rule, env))
            
            connection      = SimpleConnection(lr_list, env, layer.input_size, layer.output_size, device=env.device)
            activation      = ActivationFactory.create(layer.activation.type, layer.activation, env)
            neuron_layer    = NeuronLayer(kernel, connection, activation, env)
            
            network.add_layer(neuron_layer, layer.name)
            
        network.to(env.device)
        
        return network

    