"""
Here we visualize our data and results
"""
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

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

from network.loss.binary_loss import BinaryLoss

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
            ATTR(MODEL_NS.DATASET_SIZE),
            DataType.TRAIN,
            OutputType.TORCH, 
            LatencyEncoder(1),
            os.path.join("data", "data", "random"))
    
    
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
        data, label = self._dataset[IDX]
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
            data, label = self._dataset[IDX]
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
        data, _ = self._dataset[IDX]
        data = {batch_name: data}
        
        input_layer = DENNode(ATTR(MODEL_NS.NUM_INPUTS))
        output_layer = Node(ATTR(MODEL_NS.NUM_OUTPUTS))
        connection = Connection(input_layer, output_layer)
        
        network = Network(ATTR(MODEL_NS.NUM_OUTPUTS), False)
        
        network.add_layer(input_layer, Network.INPUT_LAYER_NAME)
        network.add_layer(output_layer, Network.OUTPUT_LAYER_NAME)
        
        network.add_connection(connection, Network.INPUT_LAYER_NAME, Network.OUTPUT_LAYER_NAME)
        
        response = network.run(data)
        response = response[batch_name]
        
        # Plot the results
        plt.figure()

        plt.plot(response.detach().numpy().flatten(), label='Model Output Voltage', color='red')
        plt.title('Output of a simple one layer fully connected network')
        plt.xlabel('Time Steps')
        plt.ylabel('Voltage')
        plt.legend()

        plt.tight_layout()
        plt.show()
        
    def train_max_time(self):
        """
        train the a simple one layer fully connected over random single spike data
        """
        
        input_layer = DENNode(ATTR(MODEL_NS.NUM_INPUTS))
        output_layer = SingleSpikeNode(ATTR(MODEL_NS.NUM_OUTPUTS))
        connection = Connection(input_layer, output_layer)
        
        network = Network(ATTR(MODEL_NS.BATCH_SIZE), False)
        
        network.add_layer(input_layer, Network.INPUT_LAYER_NAME)
        network.add_layer(output_layer, Network.OUTPUT_LAYER_NAME)
        
        network.add_connection(connection, Network.INPUT_LAYER_NAME, Network.OUTPUT_LAYER_NAME)
        
        criterion = BinaryLoss
        
        optimizer = torch.optim.SGD(network.parameters(), lr=0.01)
        num_epochs = 10
        
        dataloader = torch.utils.data.DataLoader(self._dataset, batch_size=ATTR(MODEL_NS.BATCH_SIZE), shuffle=True)

        for epoch in range(num_epochs):
            running_loss = 0.0
            # Use tqdm to display progress during each epoch
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch [{epoch+1}/{num_epochs}]")
            
            for i, (inputs, labels) in progress_bar:
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = network.forward(inputs)

                # Calculate loss
                loss = criterion.apply(outputs, labels.unsqueeze(1).float())

                # Backward pass (autograd is triggered here)
                loss.backward()

                # Update the weights
                optimizer.step()

                # Update the running loss
                running_loss += loss.item()

                # Update progress bar with loss
                progress_bar.set_postfix(loss=running_loss / (i + 1))