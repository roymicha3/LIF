"""
Here we visualize our data and results
"""
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from ray import tune

from encoders.spike.latency_encoder import LatencyEncoder
from data.dataset.random_dataset import RandomDataset, DataType, OutputType
from common import Configuration, SPIKE_NS, MODEL_NS, DATA_NS

from network.nodes.node import Node
from network.nodes.leaky_node import LeakyNode
from network.nodes.den_node import DENNode
from network.nodes.single_spike_node import SingleSpikeNode
from network.topology.connection import Connection
from network.topology.network import Network
from network.learning.optimizers import MomentumOptimizer

from network.loss.binary_loss import BinaryLoss

class RandomSpikePattern:
    """
    visualize the random spike pattern data
    """
    def __init__(self, config: Configuration) -> None:
        self.config = config
        
        self._dataset = RandomDataset(
            self.config,
            DataType.TRAIN,
            OutputType.TORCH, 
            LatencyEncoder(self.config, 1))
    
    
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
        
        kernel = LeakyNode(self.config, n, scale=True)
        
        with torch.no_grad():
            response = kernel.forward(data)

        simulated_response = LeakyNode.assimulate_response(raw_data, self.config[SPIKE_NS.tau], self.config[SPIKE_NS.dt])
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
            
            kernel = DENNode(self.config, n, scale=True)
            
            with torch.no_grad():
                response = kernel.forward(data)

            simulated_response = DENNode.assimulate_response(raw_data, self.config[SPIKE_NS.tau_m], self.config[SPIKE_NS.tau_s], self.config[SPIKE_NS.dt])
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
        
        input_layer = DENNode(self.config, self.config[MODEL_NS.NUM_INPUTS])
        output_layer = Node(self.config[MODEL_NS.NUM_OUTPUTS])
        connection = Connection(input_layer, output_layer)
        
        network = Network(self.config, False)
        
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
        Train the a simple one layer fully connected over random single spike data.

        This function trains the network and also calculates the training accuracy per epoch.
        """

        input_layer = DENNode(self.config, self.config[MODEL_NS.NUM_INPUTS])
        output_layer = SingleSpikeNode(self.config, self.config[MODEL_NS.NUM_OUTPUTS])
        connection = Connection(input_layer, output_layer)#, wmin=0, wmax=1)

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

            # Use tqdm to display progress during each epoch
            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch [{epoch+1}/{num_epochs}]")

            for i, (inputs, labels) in progress_bar: 
                optimizer.zero_grad()
                # Forward pass
                outputs = network.forward(inputs)

                # Calculate loss
                loss = criterion.forward(outputs, labels.unsqueeze(1).float())

                # Backward pass
                network.backward(criterion.backward())
                optimizer.step()

                # Update the running loss
                running_loss += torch.sum(loss)

                # Calculate predictions and update accuracy
                predicted = (outputs.squeeze() > 0) * 2 - 1
                # labels = labels.squeeze()
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

                # Update progress bar with loss and accuracy
                accuracy = 100 * correct_predictions / total_predictions
                progress_bar.set_postfix(loss=running_loss, accuracy=accuracy)

            # Print epoch summary (optional)
            print(f"[Epoch {epoch + 1}] Loss: {running_loss / len(dataloader):.3f}, Accuracy: {accuracy:.2f}%")


    # def tune_model(self, config):
    #     input_layer = DENNode(self.config[MODEL_NS.NUM_INPUTS])
    #     output_layer = SingleSpikeNode(self.config[MODEL_NS.NUM_OUTPUTS])
    #     connection = Connection(input_layer, output_layer)#, wmin=0, wmax=1)

    #     network = Network(config["batch_size"], False)

    #     network.add_layer(input_layer, Network.INPUT_LAYER_NAME)
    #     network.add_layer(output_layer, Network.OUTPUT_LAYER_NAME)

    #     network.add_connection(connection, Network.INPUT_LAYER_NAME, Network.OUTPUT_LAYER_NAME)
    #     optimizer = MomentumOptimizer(connection.parameters(), lr=config["lr"], momentum=config["momentum"])

    #     criterion = BinaryLoss()

    #     num_epochs = 500

    #     dataloader = torch.utils.data.DataLoader(self._dataset, batch_size=config["batch_size"], shuffle=True)

    #     for epoch in range(num_epochs):
    #         running_loss = 0.0
    #         correct_predictions = 0
    #         total_predictions = 0

    #         # Use tqdm to display progress during each epoch
    #         progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch [{epoch+1}/{num_epochs}]")

    #         for i, (inputs, labels) in progress_bar: 
    #             optimizer.zero_grad()
                
    #             # Forward pass
    #             outputs = network.forward(inputs)

    #             # Calculate loss
    #             loss = criterion.forward(outputs, labels.unsqueeze(1).float())

    #             # Backward pass
    #             network.backward(criterion.backward())
    #             optimizer.step()

    #             # Update the running loss
    #             running_loss += torch.sum(loss)

    #             # Calculate predictions and update accuracy
    #             predicted = (outputs.squeeze() > 0) * 2 - 1
    #             # labels = labels.squeeze()
    #             correct_predictions += (predicted == labels).sum().item()
    #             total_predictions += labels.size(0)

    #             # Update progress bar with loss and accuracy
    #             accuracy = 100 * correct_predictions / total_predictions
    #             progress_bar.set_postfix(loss=running_loss, accuracy=accuracy)

    #         # Print epoch summary (optional)
    #         print(f"[Epoch {epoch + 1}] Loss: {running_loss / len(dataloader):.3f}, Accuracy: {accuracy:.2f}%")

    #         tune.report(loss=running_loss, accuracy=accuracy)

    # def run_tune_model(self):
    #     # Define the search space
    #     search_space = {
    #         "lr": tune.loguniform(1e-5, 1e-1),
    #         "momentum": tune.loguniform(1e-2, 0.999),
    #         "batch_size": tune.randint(1, 32),
    #     }
        
    #     run = lambda config: RandomSpikePattern.tune_model(self, config)
    #     # Launch the tuning experiment
    #     with tune.run(run, config=search_space, num_samples=10):
    #         # Analyze the results
    #         best_trial = tune.experiment.get_best_trial("accuracy", mode="max")
    #         best_config = best_trial.config
    #         best_accuracy = best_trial.metric_analysis["accuracy"]["max"]

    #         print("Best config:", best_config)
    #         print("Best accuracy:", best_accuracy)