from typing import Optional, Dict, Any
import mlflow
import os

from pipeline.callback.callback import Callback, Metrics


class MlflowCallback(Callback):
    def __init__(self, experiment_name: Optional[str] = None, 
                 experiment_dir: Optional[str] = None, 
                 run_dir: Optional[str] = None):
        """
        Initialize the MlflowCallback.

        Args:
            experiment_name (str, optional): Name of the MLflow experiment to log metrics to.
                                             If provided, sets or gets the experiment.
            experiment_dir (str, optional): Directory to store experiment artifacts.
            run_dir (str, optional): Directory to store run artifacts.
        """
        if experiment_name:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                if experiment_dir:
                    artifact_location = os.path.abspath(experiment_dir)
                    mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
                else:
                    mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        
        if run_dir:
            self.active_run = mlflow.start_run(artifact_location=os.path.abspath(run_dir))
        else:
            self.active_run = mlflow.start_run()

    def on_epoch_end(self, metrics: Dict[str, Any]) -> bool:
        """
        Logs metrics to MLflow at the end of each epoch.
        """
        for key, value in metrics.items():
            if isinstance(key, Metrics):
                mlflow.log_metric(key.value, value)
            else:
                mlflow.log_metric(key, value)
        return True

    def on_train_end(self, metrics: Dict[str, Any]):
        """
        Logs final metrics to MLflow at the end of training and ends the run.
        """
        self.on_epoch_end(metrics)  # Log final metrics
        mlflow.end_run()

    def __del__(self):
        """
        Ensure the run is ended if the callback is deleted.
        """
        if mlflow.active_run():
            mlflow.end_run()
