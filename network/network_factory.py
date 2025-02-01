from network.topology.network import Network
from network.topology.neuron import NeuronLayer
from network.kernel.den_kernel import DENKernel
from network.learning.single_spike_lr import SingleSpikeLR
from network.topology.fully_connected_connection import SimpleConnection

from common import SPIKE_NS, MODEL_NS, DATA_NS

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
        n = config[MODEL_NS.NUM_INPUTS]
        kernel = DENKernel(config, n, device=device)
        lr = SingleSpikeLR(config)
        connection = SimpleConnection(lr, (n, 1), device=device)
        
        layer = NeuronLayer(config, kernel, connection)

        # Create the network and add layers and connection
        network = Network(config[DATA_NS.BATCH_SIZE], device=device)
        network.add_layer(layer, "Input")
        
        return network
