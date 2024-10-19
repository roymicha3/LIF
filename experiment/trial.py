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

import numpy as np

class Trial:
    """
    A class to run a trial of a spiking neural network with a defined configuration.
    """

    @staticmethod
    def run(config: Configuration, report: bool = True, early_stopping_patience: int = 15):
        """
        Run the trial using the specified configuration with early stopping.

        :param config: A Configuration object that holds the parameters for the trial.
        :param report: Whether to report metrics for Ray Tune.
        :param early_stopping_patience: Number of epochs to wait for improvement before stopping early.
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
        connection = Connection(input_layer, output_layer, bias=False)

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
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
        
        valid_size = int(config[DATA_NS.DATASET_SIZE] * (config[DATA_NS.VALIDATION_PERCENTAGE] / 100))
        valid_dataset = torch.utils.data.Subset(dataset, np.arange(1, valid_size))
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)#config["batch_size"])
        
        # Early stopping variables
        best_loss = np.inf
        patience_counter = 0

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_predictions = 0

            # Training loop
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

                # Update running loss and accuracy
                running_loss += torch.sum(loss)
                predicted = criterion.classify(outputs)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

                # Update progress bar with loss and accuracy
                accuracy = 100 * correct_predictions / total_predictions
                progress_bar.set_postfix(loss=running_loss.item(), accuracy=accuracy)

            # Compute full dataset loss and accuracy after each epoch
            total_loss, total_accuracy = Trial.evaluate(network, criterion, valid_dataloader)

            # Print epoch summary
            print(f"[Epoch {epoch + 1}] Loss: {total_loss:.3f}, Accuracy: {total_accuracy:.2f}%")

            # Report results to Ray Tune if required
            if report:
                train.report({"loss": total_loss, "accuracy": total_accuracy})

            # Early stopping logic
            if total_loss < best_loss:
                best_loss = total_loss
                patience_counter = 0  # Reset patience if the loss improves
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
                break
            
            if total_accuracy >= 99.9:
                print(f"Early stopping at epoch {epoch + 1} due to 100% train accuracy.")
                break

    @staticmethod
    def evaluate(network, criterion, dataloader):
        """
        Compute the loss and accuracy over the entire dataset.

        :param network: The spiking neural network.
        :param criterion: The loss function.
        :param dataloader: The DataLoader for the dataset.
        :return: Tuple of (loss, accuracy).
        """
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for inputs, labels in dataloader:
                outputs = network.forward(inputs)
                loss = criterion.forward(outputs, labels.unsqueeze(1).float())
                total_loss += torch.sum(loss)

                predicted = criterion.classify(outputs)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

        average_loss = total_loss / len(dataloader)
        accuracy = 100 * correct_predictions / total_predictions
        return average_loss, accuracy
