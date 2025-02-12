from omegaconf import DictConfig

from settings.factory import Factory
from settings.serializable import YAMLSerializable

from network.kernel.kernel_factory import KernelFactory
from network.learning.lr_factory import LearningRuleFactory

from network.topology.network import Network
from network.topology.neuron import NeuronLayer
from network.topology.fully_connected_connection import SimpleConnection

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
    def build_network(config: DictConfig, env_config: DictConfig) -> Network:
        """
        builds a network out of a config file
        """
        network = Network(config, learning=True, device=env_config.device)
        
        for layer in config.layers:
            kernel = KernelFactory.create(layer.kernel.type, layer.kernel, env_config)
            learning_rule = LearningRuleFactory.create(layer.learning_rule.type, layer.learning_rule, env_config)
            connection = SimpleConnection(learning_rule, layer.input_size, layer.output_size, device=env_config.device)
            neuron_layer = NeuronLayer(kernel, connection)
            network.add_layer(neuron_layer, layer.name)
            
        network.to(env_config.device)
        
        return network
    
    @staticmethod
    def create(name: str, config: DictConfig, env_config: DictConfig):
        return NetworkFactory.build_network(config, env_config)

    