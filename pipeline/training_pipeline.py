import torch
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import numpy as np

from network.network_factory import NetworkFactory
from network.optimizer.optimizer_factory import OptimizerFactory
from network.lr_scheduler.lr_scheduler_factory import LRSchedulerFactory
from network.loss.loss_factory import LossFactory
from pipeline.pipline import Pipeline
from pipeline.callback.callback import Metric
from pipeline.callback.callback_factory import CallbackFactory
from settings.serializable import YAMLSerializable


class TrainingPipeline(Pipeline, YAMLSerializable):
    """
    class that is responsible for the training of the Network
    """
    def __init__(self, 
                 epochs: int, 
                 batch_size: int, 
                 validation_split: float, 
                 test_split: float,
                 shuffle: bool = True):
        
        super(TrainingPipeline, self).__init__()
        super(YAMLSerializable, self).__init__()
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.shuffle = shuffle
        
    @classmethod
    def from_config(cls, config: DictConfig, env_config: DictConfig):
        pipeline = cls(
            config.epochs,
            config.batch_size, 
            config.validation_split,
            config.test_split,
            config.shuffle)
        
        for callback_config in config.callbacks:
            callback = CallbackFactory.create(callback_config.type, callback_config, env_config)
            pipeline.register_callback(callback)
            
        return pipeline
    
    
    def run(self, config: DictConfig, env_config: DictConfig):
        """
        Train the model using the provided data loader.
        """
        dataset = self.load_dataset(config.dataset, env_config)
        val_size = int(self.validation_split * len(dataset))
        val_dataset = torch.utils.data.Subset(dataset, np.arange(1, val_size))
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle)
        
        network = NetworkFactory.create(config.model.type, config.model, env_config)
        
        optimizer = OptimizerFactory.create(
            config.optimizer.type,
            config.optimizer,
            network.parameters())
        
        scheduler = LRSchedulerFactory.create(config.lr_scheduler.type, optimizer, config.lr_scheduler)
        criterion = LossFactory.create(config.loss.type, config.loss, env_config)
        
        for epoch in range(self.epochs):
            correct_predictions = 0
            total_predictions = 0
 
            # Training loop
            progress_bar = tqdm(
                enumerate(dataloader),
                total=len(dataloader),
                desc=f"Epoch [{epoch+1}/{self.epochs}]")
            
            for _, (inputs, labels) in progress_bar:
                inputs = inputs.to(env_config.device)  # Move inputs to the correct device
                labels = labels.to(env_config.device)  # Move labels to the correct device

                # Forward pass
                outputs = network.forward(inputs)

                # Calculate loss
                loss = criterion.forward(outputs, labels.unsqueeze(1).float())

                # Backward pass
                network.backward(criterion.backward())
                optimizer.step()
                scheduler.step()

                # Update running loss and accuracy
                running_loss = torch.sum(loss).item()
                predicted = criterion.classify(outputs)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

                # Update progress bar with loss and accuracy
                accuracy = 100 * correct_predictions / total_predictions
                progress_bar.set_postfix(loss=running_loss, accuracy=accuracy)

            torch.cuda.empty_cache()
            
            # Compute full dataset loss and accuracy after each epoch
            total_loss, total_accuracy = self.evaluate(network, criterion, val_dataset)
            
            epoch_res = \
                {
                    Metric.VAL_LOSS: total_loss,
                    Metric.VAL_ACC: total_accuracy,
                    Metric.NETWORK: network
                }
            
            stop_flag = self.on_epoch_end(epoch_res)
            if stop_flag:
                print("A callback issued a stop! \n")
                break
            
            # Print epoch summary
            print(f"[Epoch {epoch + 1}] Loss: {total_loss:.3f}, Accuracy: {total_accuracy:.2f}%")

            if total_accuracy >= 99.9:
                print(f"Early stopping at epoch {epoch + 1} due to 100% train accuracy.")
                break
        
        self.on_end({})

    def evaluate(self, network, criterion, dataset):
        """
        Compute the loss, overall accuracy, and accuracy per label type over the entire dataset.

        :param network: The spiking neural network.
        :param criterion: The loss function.
        :param dataloader: The DataLoader for the dataset.
        :return: Tuple of (loss, overall accuracy, accuracy per label type).
        """
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=64)
        
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        label_correct = {}
        label_total = {}

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(network.device)  # Ensure inputs are moved to the correct device
                labels = labels.to(network.device)  # Move labels to the correct device

                # Forward pass
                outputs = network.forward(inputs)
                loss = criterion.forward(outputs, labels.unsqueeze(1).float())
                total_loss += torch.sum(loss).item()

                # Classify the outputs
                predicted = criterion.classify(outputs)

                # Update overall correct predictions and total predictions
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

                # Update per-label correct and total counts
                for label in torch.unique(labels):  # Iterate over all unique labels

                    # Ensure we index correctly, handling batch-wise dimensions
                    label_correct[label.item()] = label_correct.get(label.item(), 0) + (predicted[labels == label] == labels[labels == label]).sum().item()
                    label_total[label.item()] = label_total.get(label.item(), 0) + (labels == label).sum().item()

        # Calculate average loss and overall accuracy
        average_loss = total_loss / len(dataloader)
        accuracy = 100 * correct_predictions / total_predictions

        # Calculate per-label accuracy
        accuracy_per_label = {}
        for label in label_correct:
            accuracy_per_label[label] = 100 * label_correct[label] / label_total[label]
            print(f"The accuracy for label: {label} is: {accuracy_per_label[label]:.2f}%")

        return average_loss, accuracy
    

    def save(self, file_path):
        """
        Save the training pipeline configuration to YAML.
        """
        config = DictConfig({
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'validation_split': self.validation_split,
            'test_split': self.test_split,
            'shuffle': self.shuffle
        })
        with open(file_path, 'w', encoding='utf-8') as f:
            OmegaConf.save(config, f)