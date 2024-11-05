"""
Visualization module for spike data and network responses.
"""

import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from ray import tune

from encoders.spike.latency_encoder import LatencyEncoder
from data.dataset.random_dataset import RandomDataset, DataType, OutputType
from common import Configuration, SPIKE_NS, MODEL_NS, DATA_NS
from network.kernel.kernel import Kernel
from network.kernel.leaky_kernel import LeakyKernel
from network.kernel.den_kernel import DENKernel
from network.kernel.single_spike_node import SingleSpikeNode
from network.topology.connection import Connection
from network.topology.network import Network
from network.learning.optimizers import MomentumOptimizer
from network.loss.binary_loss import BinaryLoss

class RandomSpikePattern:
    """
    Visualize and analyze random spike pattern data.
    """

    def __init__(self, config: Configuration) -> None:
        """
        Initialize the RandomSpikePattern class.

        Args:
            config (Configuration): Configuration object containing model parameters.
        """
        self.config = config
        self._dataset = RandomDataset(
            self.config,
            DataType.TRAIN,
            OutputType.TORCH, 
            LatencyEncoder(self.config, 1)
        )
    
    def single_spike_raster(self):
        """
        Plot rasters of the random spike patterns.
        """
        FIRST_IDX = 0
        data = self._dataset.get_raw(FIRST_IDX)
        data.plot()
        
    def lif_response(self):
        """
        Plot the response of a Leaky Integrate-and-Fire (LIF) neuron to spike input.
        """
        IDX = 1
        data, _ = self._dataset[IDX]
        raw_data = self._dataset.get_raw(IDX)
        
        seq_len, n = data.shape
        kernel = LeakyKernel(self.config, n, scale=True)
        
        with torch.no_grad():
            response = kernel.forward(data)

        simulated_response = LeakyKernel.assimulate_response(raw_data, self.config[SPIKE_NS.tau], self.config[SPIKE_NS.dt])
        simulated_response = simulated_response.numpy()
        
        self._plot_response(response, simulated_response, "LIF Neuron Response")
        
    def den_response(self):
        """
        Plot the response of a Dual Exponential Neuron (DEN) to spike input.
        """
        IDX = 2
        data, _ = self._dataset[IDX]
        raw_data = self._dataset.get_raw(IDX)
        
        seq_len, n = data.shape
        kernel = DENKernel(self.config, n, scale=True)
        
        with torch.no_grad():
            response = kernel.forward(data)

        simulated_response = DENKernel.assimulate_response(
            raw_data, 
            self.config[SPIKE_NS.tau_m], 
            self.config[SPIKE_NS.tau_s], 
            self.config[SPIKE_NS.dt]
        )
        simulated_response = simulated_response.numpy()
        
        self._plot_response(response, simulated_response, "DEN Neuron Response")
        
    def simple_network_response(self):
        """
        Plot the response for a random spike input of a single layer single spike network.
        """
        IDX = 3
        batch_name = f"{IDX}"
        data, _ = self._dataset[IDX]
        data = {batch_name: data}
        
        input_layer = DENKernel(self.config, self.config[MODEL_NS.NUM_INPUTS])
        output_layer = Kernel(self.config[MODEL_NS.NUM_OUTPUTS])
        connection = Connection(input_layer, output_layer)
        
        network = Network(self.config, False)
        network.add_layer(input_layer, Network.INPUT_LAYER_NAME)
        network.add_layer(output_layer, Network.OUTPUT_LAYER_NAME)
        network.add_connection(connection, Network.INPUT_LAYER_NAME, Network.OUTPUT_LAYER_NAME)
        
        response = network.run(data)
        response = response[batch_name]
        
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
        Train a simple one layer fully connected network over random single spike data.
        This function trains the network and calculates the training accuracy per epoch.
        """
        input_layer = DENKernel(self.config, self.config[MODEL_NS.NUM_INPUTS])
        output_layer = SingleSpikeNode(self.config, self.config[MODEL_NS.NUM_OUTPUTS])
        connection = Connection(input_layer, output_layer)

        network = Network(self.config, False)
        network.add_layer(input_layer, Network.INPUT_LAYER_NAME)
        network.add_layer(output_layer, Network.OUTPUT_LAYER_NAME)
        network.add_connection(connection, Network.INPUT_LAYER_NAME, Network.OUTPUT_LAYER_NAME)
        
        optimizer = MomentumOptimizer(connection.parameters(), lr=self.config[MODEL_NS.LR], momentum=0.9)
        criterion = BinaryLoss()
        num_epochs = 500
        dataloader = torch.utils.data.DataLoader(self._dataset, batch_size=self.config[DATA_NS.BATCH_SIZE], shuffle=True)

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch [{epoch+1}/{num_epochs}]")

            for i, (inputs, labels) in progress_bar: 
                optimizer.zero_grad()
                outputs = network.forward(inputs)
                loss = criterion.forward(outputs, labels.unsqueeze(1).float())
                network.backward(criterion.backward())
                optimizer.step()

                running_loss += torch.sum(loss)
                predicted = (outputs.squeeze() > 0) * 2 - 1
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

                accuracy = 100 * correct_predictions / total_predictions
                progress_bar.set_postfix(loss=running_loss, accuracy=accuracy)

            print(f"[Epoch {epoch + 1}] Loss: {running_loss / len(dataloader):.3f}, Accuracy: {accuracy:.2f}%")

    def _plot_response(self, response, simulated_response, title):
        """
        Helper method to plot neuron responses.

        Args:
            response (torch.Tensor): Actual response from the model.
            simulated_response (numpy.ndarray): Simulated response.
            title (str): Title for the plot.
        """
        plt.figure()
        plt.plot(response, label='Output Spikes', color='red')
        plt.plot(simulated_response, label='Simulated Output Spikes', color='blue', linestyle=":")
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Spike')
        plt.legend()
        plt.tight_layout()
        plt.show()
