import torch
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import numpy as np

from experiment.db.database import DB


from network.network_factory import NetworkFactory
from network.optimizer.optimizer_factory import OptimizerFactory
from network.lr_scheduler.lr_scheduler_factory import LRSchedulerFactory
from network.loss.loss_factory import LossFactory
from pipeline.pipeline import Pipeline
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
                 shuffle: bool = True,
                 id: int = None):
        
        super(TrainingPipeline, self).__init__()
        super(YAMLSerializable, self).__init__()
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.shuffle = shuffle
        self.id = id
        
    @classmethod
    def from_config(cls, config: DictConfig, env_config: DictConfig, id: int = None):
        pipeline = cls(
            config.epochs,
            config.batch_size, 
            config.validation_split,
            config.test_split,
            config.shuffle,
            id)
        
        for callback_config in config.callbacks:
            callback = CallbackFactory.create(callback_config.type, callback_config, env_config, id)
            pipeline.register_callback(callback)
            
        return pipeline
    
    
    def run(self, config: DictConfig, env_config: DictConfig):
        """
        Train the model using the provided data loader.
        """
        dataset = self.load_dataset(config.dataset, env_config)
        val_size = int(self.validation_split * len(dataset))
        val_dataset = torch.utils.data.Subset(dataset, np.arange(1, val_size))
        
        
        network = NetworkFactory.create(config.model.type, config.model, env_config)
        
        optimizer = OptimizerFactory.create(
            config.optimizer.type,
            config.optimizer,
            network.parameters())
        
        scheduler = LRSchedulerFactory.create(config.lr_scheduler.type, optimizer, config.lr_scheduler)
        criterion = LossFactory.create(config.loss.type, config.loss, env_config)
        
        status = "completed"
        
        for epoch in range(self.epochs):
            
            indices = torch.randperm(len(dataset))
            dataset = torch.utils.data.Subset(dataset, indices)
            
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle)
            
            DB.instance().create_epoch(self.id, epoch)
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
                

                # Update running loss and accuracy
                running_loss = torch.sum(loss).item()
                predicted = criterion.classify(outputs)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

                # Update progress bar with loss and accuracy
                accuracy = 100 * correct_predictions / total_predictions
                progress_bar.set_postfix(loss=running_loss, accuracy=accuracy)

            
            scheduler.step()
            torch.cuda.empty_cache()
            
            # Compute full dataset loss and accuracy after each epoch
            total_loss, total_accuracy = self.evaluate(network, criterion, val_dataset)
            
            epoch_res = \
                {
                    Metric.VAL_LOSS: total_loss,
                    Metric.VAL_ACC: total_accuracy,
                    Metric.NETWORK: network
                }
            
            stop_flag = self.on_epoch_end(epoch, epoch_res)
            if stop_flag:
                status = "stopped"
                print("A callback issued a stop! \n")
                break
            
            # Print epoch summary
            print(f"[Epoch {epoch + 1}] Loss: {total_loss:.3f}, Accuracy: {total_accuracy:.2f}%")

            if total_accuracy >= 99.9:
                print(f"Early stopping at epoch {epoch + 1} due to 100% train accuracy.")
                break
        
        self.on_end({Metric.STATUS: status})

    def evaluate(self, network, criterion, dataset):
        """
        Compute the loss, overall accuracy, and accuracy per label type over the entire dataset.

        Args:
            network: The spiking neural network
            criterion: The loss function
            dataset: The dataset to evaluate on
            
        Returns:
            Tuple of (average_loss, overall_accuracy)
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
                inputs = inputs.to(network.device)
                labels = labels.to(network.device)

                # Forward pass
                outputs = network.forward(inputs)
                loss = criterion.forward(outputs, labels.unsqueeze(1).float())
                total_loss += torch.sum(loss).item()

                # Classify the outputs
                predicted = criterion.classify(outputs)
                
                # Ensure predicted has the same shape as labels
                if predicted.dim() == 0:
                    predicted = predicted.unsqueeze(0)  # Make it 1D if it's a scalar
                
                # Make sure both tensors are 1D
                predicted = predicted.view(-1)
                labels = labels.view(-1)

                # Update overall accuracy metrics
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += labels.size(0)

                # Update per-label accuracy metrics
                unique_labels = torch.unique(labels)
                for label in unique_labels:
                    label_val = label.item()
                    label_mask = (labels == label_val)
                    
                    # Get current counts or initialize to 0
                    current_correct = label_correct.get(label_val, 0)
                    current_total = label_total.get(label_val, 0)
                    
                    # Update counts - ensure we're comparing properly shaped tensors
                    matches = (predicted[label_mask] == label_val).sum().item()
                    total = label_mask.sum().item()
                    
                    label_correct[label_val] = current_correct + matches
                    label_total[label_val] = current_total + total

        # Calculate average loss and overall accuracy
        average_loss = total_loss / len(dataloader)
        accuracy = 100 * correct_predictions / total_predictions

        # Calculate and print per-label accuracy
        for label_val in label_correct:
            if label_total[label_val] > 0:  # Avoid division by zero
                label_accuracy = 100 * label_correct[label_val] / label_total[label_val]
                print(f"Accuracy for label {label_val}: {label_accuracy:.2f}%")

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