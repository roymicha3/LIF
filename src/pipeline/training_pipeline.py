import torch
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from experiment_manager.common.common import Metric
from experiment_manager.common.common import RunStatus
from experiment_manager.environment import Environment
from experiment_manager.pipelines.pipeline import Pipeline
from experiment_manager.common.serializable import YAMLSerializable

from network.loss.loss_factory import LossFactory
from network.network_factory import NetworkFactory
from network.optimizer.optimizer_factory import OptimizerFactory
from network.lr_scheduler.lr_scheduler_factory import LRSchedulerFactory

from encoders.encoder_factory import EncoderFactory
from data.dataset.dataset_factory import DatasetFactory
from data.dataset.dataset import Dataset, DataType, OutputType

@YAMLSerializable.register("TrainingPipeline")
class TrainingPipeline(Pipeline, YAMLSerializable):
    """
    class that is responsible for the training of the Network
    """
    def __init__(self,
                 env: Environment,
                 epochs: int, 
                 batch_size: int, 
                 validation_split: float, 
                 test_split: float,
                 shuffle: bool = True,
                 id: int = None):
        
        super(TrainingPipeline, self).__init__(env)
        super(YAMLSerializable, self).__init__()
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.test_split = test_split
        self.shuffle = shuffle
        self.id = id
        
    @classmethod
    def from_config(cls, config: DictConfig, env: Environment, id: int = None):
        pipeline = cls(
            env,
            config.pipeline.epochs,
            config.pipeline.batch_size, 
            config.pipeline.validation_split,
            config.pipeline.test_split,
            config.pipeline.shuffle,
            id)
            
        return pipeline
    
    def load_dataset(self, 
                     dataset_config: DictConfig, 
                     env: Environment, 
                     type_: DataType = DataType.TRAIN) -> Dataset:
        
        encoder_config = dataset_config.encoder
        encoder = EncoderFactory.create(encoder_config.type, encoder_config, env) # TODO: fix this to accept environment
        dataset = DatasetFactory.create(
            dataset_config.type, 
            dataset_config, 
            type_, 
            OutputType.TORCH, 
            encoder)
        
        return dataset
    
    @Pipeline.epoch_wrapper
    def run_epoch(self, epoch_idx, model, *args, **kwargs): # TODO: update the functions signature
        correct_predictions = 0
        total_predictions = 0
        
        train_dataloader    = kwargs["train_dataloader"]
        val_dataloader      = kwargs["val_dataloader"]
        criterion       = kwargs["criterion"]
        optimizer       = kwargs["optimizer"]
        scheduler       = kwargs["scheduler"]
        device          = kwargs["device"]
        
        # Training loop
        progress_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch [{epoch_idx+1}/{self.epochs}]")
        
        for _, (inputs, labels) in progress_bar:
            inputs = inputs.to(self.env.device)
            labels = labels.to(self.env.device)
            
            # Forward pass
            outputs = model.forward(inputs)

            # Calculate loss
            loss = criterion.forward(outputs, labels.unsqueeze(1).float())

            # Backward pass
            model.backward(criterion.backward())
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
        # torch.cuda.empty_cache() # TODO: check if this is needed
        
        # Compute full dataset loss and accuracy after each epoch
        # total_train_loss, total_train_accuracy = self.evaluate(model, criterion, train_dataloader)
        total_val_loss, total_val_accuracy = self.evaluate(model, criterion, val_dataloader)
        
        self.epoch_metrics = \
            {
                # Metric.TRAIN_LOSS: total_train_loss,
                # Metric.TRAIN_ACC: total_train_accuracy,
                Metric.VAL_LOSS: total_val_loss,
                Metric.VAL_ACC: total_val_accuracy,
                Metric.NETWORK: model
            }
        
        # Print epoch summary
        self.env.logger.info(f"[Epoch {epoch_idx + 1}] Loss: {total_val_loss:.3f}, Accuracy: {total_val_accuracy:.2f}%")

        if total_val_accuracy >= 99.9:
            self.env.logger.info(f"Early stopping at epoch {epoch_idx + 1} due to 100% train accuracy.")
            return RunStatus.SUCCESS
        
        return RunStatus.FINISHED
    
    
    @Pipeline.run_wrapper
    def run(self, config: DictConfig):
        """
        Train the model using the provided data loader.
        """
        status = RunStatus.SKIPPED
        dataset = self.load_dataset(config.dataset, self.env)
        val_size = int(self.validation_split * len(dataset))
        val_dataset = torch.utils.data.Subset(dataset, np.arange(1, val_size))
        
        
        network = NetworkFactory.create(
            config.model.type,
            config.model,
            self.env) # TODO: fix to accept environment
        
        optimizer = OptimizerFactory.create(
            config.optimizer.type,
            config.optimizer,
            network.parameters())
        
        scheduler = LRSchedulerFactory.create(
            config.lr_scheduler.type,
            optimizer,
            config.lr_scheduler)
        
        criterion = LossFactory.create(
            config.loss.type,
            config.loss,
            self.env)
        
        for epoch in range(self.epochs):
            
            indices = torch.randperm(len(dataset))
            dataset = torch.utils.data.Subset(dataset, indices)
            
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=self.shuffle)
            
            val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                         batch_size=self.batch_size,
                                                         shuffle=False)
            
            status = self.run_epoch(
                epoch, 
                network, 
                train_dataloader = dataloader, 
                val_dataloader = val_dataloader,
                criterion = criterion, 
                optimizer = optimizer,
                scheduler = scheduler,
                device = self.env.device)
            
            if status == RunStatus.SUCCESS:
                break
        
        final_accuracy, final_loss = self.evaluate(network, criterion, dataloader)
         
        self.run_metrics = \
            {
                Metric.NETWORK: network,
                Metric.TEST_ACC: final_accuracy,
                Metric.TEST_LOSS: final_loss,
            }
            
        return status
    
    def evaluate(self, network, criterion, dataloader): # TODO: might be a good idea to add a per label accuracy
        """
        Compute the loss, overall accuracy, and accuracy per label type over the entire dataset.
            
        Returns:
            Tuple of (average_loss, overall_accuracy)
        """
        
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