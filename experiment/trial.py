import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from ray import train, tune

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


class Trial:
    """
    A class to run a trial of a spiking neural network with a defined configuration.
    """

    @staticmethod
    def run(config: Configuration, report: bool = True):
        """
        Run the trial using the specified configuration.

        :param config: A Configuration object that holds the parameters for the trial.
        """
        # Initialize dataset
        dataset = RandomDataset(
            config,
            DataType.TRAIN,
            OutputType.TORCH,
            LatencyEncoder(config, 1)
        )

        # Initialize layers and connection
        input_layer = DENNode(config, config[MODEL_NS.NUM_INPUTS])
        output_layer = SingleSpikeNode(config, config[MODEL_NS.NUM_OUTPUTS])
        connection = Connection(input_layer, output_layer)

        # Create a network with layers and connection
        network = Network(config[DATA_NS.BATCH_SIZE], False)
        network.add_layer(input_layer, Network.INPUT_LAYER_NAME)
        network.add_layer(output_layer, Network.OUTPUT_LAYER_NAME)
        network.add_connection(connection, Network.INPUT_LAYER_NAME, Network.OUTPUT_LAYER_NAME)

        # Set up optimizer and loss function
        # optimizer = MomentumOptimizer(
        #     connection.parameters(),
        #     lr=config["lr"],
        #     momentum=config["momentum"]
        # )
        
        optimizer = torch.optim.Adam(connection.parameters(), lr=config["lr"])
        criterion = BinaryLoss()

        num_epochs = config[MODEL_NS.EPOCHS]

        # Set up dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            # Progress bar for tracking epoch progress
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

                # Calculate predictions and accuracy
                predicted = criterion.classify(outputs)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

                # Update progress bar with loss and accuracy
                accuracy = 100 * correct_predictions / total_predictions
                progress_bar.set_postfix(loss=running_loss.item(), accuracy=accuracy)

            # Print epoch summary
            print(f"[Epoch {epoch + 1}] Loss: {running_loss / len(dataloader):.3f}, Accuracy: {accuracy:.2f}%")

            if report:
                # Report results to Ray Tune
                train.report({"loss": running_loss.item(), "accuracy": accuracy})
