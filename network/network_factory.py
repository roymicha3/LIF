from network.topology.network import Network
from network.nodes.node import Node
from network.nodes.leaky_node import LeakyNode
from network.nodes.den_node import DENNode
from network.nodes.single_spike_node import SingleSpikeNode
from network.topology.simple_connection import SimpleConnection
from network.topology.voltage_conv_connection import VoltageConvConnection
from network.nodes.exp_node import ExpNode

from common import Configuration, SPIKE_NS, MODEL_NS, DATA_NS

class NetworkFactory:
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
    def build_simple_network(config: dict, device: str) -> Network:
        """
        Constructs a network with a simple connection between input and output layers.

        Args:
            config (dict): Configuration dictionary containing model parameters.
            device (str): Device to which the network components are moved (e.g., 'cpu' or 'cuda').

        Returns:
            Network: A configured network with input and output layers connected by a SimpleConnection.
        """
        # Initialize input and output layers with specified configurations
        input_layer = DENNode(config, config[MODEL_NS.NUM_INPUTS], device=device)
        output_layer = SingleSpikeNode(config, config[MODEL_NS.NUM_OUTPUTS], device=device, learning=False)
        connection = SimpleConnection(input_layer, output_layer, device=device)

        # Create the network and add layers and connection
        network = Network(config[DATA_NS.BATCH_SIZE], device=device)
        network.add_layer(input_layer, Network.INPUT_LAYER_NAME)
        network.add_layer(output_layer, Network.OUTPUT_LAYER_NAME)
        network.add_connection(connection, Network.INPUT_LAYER_NAME, Network.OUTPUT_LAYER_NAME)
        
        for name, param in network.named_parameters():
            print(name, param.size())
        
        return network

    @staticmethod
    def build_voltage_convolution_network(config: dict, device: str) -> Network:
        """
        Constructs a network with a voltage-based convolutional connection between input and output layers.

        Args:
            config (dict): Configuration dictionary containing model parameters.
            device (str): Device to which the network components are moved (e.g., 'cpu' or 'cuda').

        Returns:
            Network: A configured network with input and output layers connected by a VoltageConvConnection.
        """
        # Initialize input and output layers with specified configurations
        input_layer = DENNode(config, config[MODEL_NS.NUM_INPUTS], device=device)
        output_layer = ExpNode(config, config[MODEL_NS.NUM_OUTPUTS], device=device, learning=True)
        connection = VoltageConvConnection(input_layer, output_layer, beta=config[MODEL_NS.BETA], device=device)

        # Create the network and add layers and connection
        network = Network(config[DATA_NS.BATCH_SIZE], device=device)
        network.add_layer(input_layer, Network.INPUT_LAYER_NAME)
        network.add_layer(output_layer, Network.OUTPUT_LAYER_NAME)
        network.add_connection(connection, Network.INPUT_LAYER_NAME, Network.OUTPUT_LAYER_NAME)
        
        return network
