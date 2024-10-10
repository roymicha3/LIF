"""
Here we visualize our data and results
"""
import matplotlib.pyplot as plt
import torch

from encoders.spike.latency_encoder import LatencyEncoder
from data.dataset.random_dataset import RandomDataset, DataType, OutputType
from tools.utils import SEQ_LEN
from common import ATTR, SPIKE_NS, MODEL_NS

from network.nodes.node import Node
from network.nodes.leaky_node import LeakyNode
from network.nodes.den_node import DENNode
from network.nodes.single_spike_node import SingleSpikeNode
from network.topology.connection import Connection
from network.topology.network import Network

DEFAULT_BATCH = 0

class RandomSpikePattern:
    """
    visualize the random spike pattern data
    """
    def __init__(self, batch_size: int = DEFAULT_BATCH) -> None:
        batch_size = batch_size if batch_size is not DEFAULT_BATCH else ATTR(MODEL_NS.BATCH_SIZE)
        
        T = ATTR(SPIKE_NS.T)
        dt = ATTR(SPIKE_NS.dt)
        self._dataset = RandomDataset(
            ATTR(MODEL_NS.NUM_INPUTS), 
            SEQ_LEN(T, dt),
            DataType.TRAIN,
            OutputType.TORCH, 
            LatencyEncoder(1))
    
    
    def single_spike_raster(self):
        """
        plots rasters of the random spike patterns
        """
        
        FIRST_IDX = 0
        data = self._dataset.get_raw(FIRST_IDX)
        
        data.plot()
        
    def lif_response(self):
        """
        plot the response of the spike
        """
        IDX = 1
        data = self._dataset[IDX]
        raw_data = self._dataset.get_raw(IDX)
        
        seq_len, n = data.shape
        
        kernel = LeakyNode(n, scale=True)
        
        with torch.no_grad():
            response = kernel.forward(data)

        simulated_response = LeakyNode.assimulate_response(raw_data, ATTR(SPIKE_NS.tau), ATTR(SPIKE_NS.dt))
        simulated_response = simulated_response.numpy()
        
        # Plot the results
        plt.figure()

        plt.plot(response, label='Output Spikes', color='red')
        plt.plot(simulated_response, label='Simulated Output Spikes', color='blue', linestyle=":")
        plt.title('Model Output Spikes')
        plt.xlabel('Time Steps')
        plt.ylabel('Spike')
        plt.legend()

        plt.tight_layout()
        plt.show()
        
    def den_response(self):
            """
            plot the response of the spike
            """
            IDX = 2
            data = self._dataset[IDX]
            raw_data = self._dataset.get_raw(IDX)
            
            seq_len, n = data.shape
            
            kernel = DENNode(n, scale=True)
            
            with torch.no_grad():
                response = kernel.forward(data)

            simulated_response = DENNode.assimulate_response(raw_data, ATTR(SPIKE_NS.tau_m), ATTR(SPIKE_NS.tau_s), ATTR(SPIKE_NS.dt))
            simulated_response = simulated_response.numpy()
            
            # Plot the results
            plt.figure()

            plt.plot(response, label='Output Spikes', color='red')
            plt.plot(simulated_response, label='Simulated Output Spikes', color='blue', linestyle=":")
            plt.title('Model Output Spikes')
            plt.xlabel('Time Steps')
            plt.ylabel('Spike')
            plt.legend()

            plt.tight_layout()
            plt.show()
        
    def simple_network_response(self):
        """
        plot the response for a random spike input of the single layer single spike network
        """
        IDX = 3
        batch_name = f"{IDX}"
        data = {batch_name: self._dataset[IDX]}
        
        input_layer = DENNode(5)
        output_layer = Node(1)
        connection = Connection(input_layer, output_layer)
        
        network = Network(1, False)
        
        network.add_layer(input_layer, Network.INPUT_LAYER_NAME)
        network.add_layer(output_layer, Network.OUTPUT_LAYER_NAME)
        
        network.add_connection(connection, Network.INPUT_LAYER_NAME, Network.OUTPUT_LAYER_NAME)
        
        response = network.run(data)
        response = response[batch_name]
        
        # Plot the results
        plt.figure()

        plt.plot(response.detach().numpy(), label='Output of a simple one layer fully connected network', color='red')
        plt.title('Model Output Voltage')
        plt.xlabel('Time Steps')
        plt.ylabel('Voltage')
        plt.legend()

        plt.tight_layout()
        plt.show()