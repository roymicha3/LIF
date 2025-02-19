from typing import Optional, Dict, Any
from omegaconf import DictConfig
import mlflow
import os

from pipeline.callback.callback import Callback, Metric, MetricCategory
from settings.serializable import YAMLSerializable

@YAMLSerializable.register("MlflowCallback")
class MlflowCallback(Callback):
    def __init__(self, experiment_name: Optional[str] = None, 
                 experiment_dir: Optional[str] = None):
        """
        Initialize the MlflowCallback.

        Args:
            experiment_name (str, optional): Name of the MLflow experiment to log metrics to.
                                             If provided, sets or gets the experiment.
            experiment_dir (str, optional): Directory to store experiment artifacts.
        """
        if experiment_name:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                if experiment_dir:
                    os.makedirs(experiment_dir, exist_ok=True)
                    artifact_location = os.path.abspath(experiment_dir)
                    mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
                else:
                    mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        
        self.active_run = mlflow.start_run()

    def on_epoch_end(self, metrics: Dict[str, Any]) -> bool:
        """
        Logs metrics to MLflow at the end of each epoch.
        """
        for key, value in metrics.items():
            try:
                metric = Metric(key)
                if metric.category == MetricCategory.TRACKED:
                    mlflow.log_metric(metric.value, value)
            except ValueError:
                # If the key is not in the Metric enum, log it as-is
                mlflow.log_metric(key, value)
        return True

    def on_train_end(self, metrics: Dict[str, Any]):
        """
        Logs final metrics to MLflow at the end of training and ends the run.
        """
        for key, value in metrics.items():
            try:
                metric = Metric(key)
                mlflow.log_metric(metric.value, value)
            except ValueError:
                # If the key is not in the Metric enum, log it as-is
                mlflow.log_metric(key, value)
        mlflow.end_run()

    def __del__(self):
        """
        Ensure the run is ended if the callback is deleted.
        """
        if mlflow.active_run():
            mlflow.end_run()
            
    @classmethod
    def from_config(cls, config: DictConfig, env_config: DictConfig):
        return cls(
            config.experiment_name,
            config.root_dir)
