"""
the main of the project
"""
import os
from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf, DictConfig
from dataclasses import dataclass

from experiment.experiment import Experiment
from experiment.db.database import DB

@dataclass
class ExperimentConfig:
    """Holds all configuration for an experiment."""
    name: str
    base_path: Path
    experiment_config: DictConfig
    model_config: DictConfig
    env_config: DictConfig
    trials_config: DictConfig

    @classmethod
    def from_directory(cls, experiment_path: str) -> 'ExperimentConfig':
        """Load all configurations from an experiment directory."""
        base_path = Path(experiment_path)
        if not base_path.exists():
            raise NotADirectoryError(f"Base directory path '{experiment_path}' does not exist.")
        
        config_path = base_path / "config"
        if not config_path.exists():
            raise NotADirectoryError(f"Config path '{config_path}' does not exist.")
        
        # Load all configurations
        experiment_config = OmegaConf.load(config_path / "experiment.yaml")
        model_config = OmegaConf.load(config_path / "config.yaml")
        env_config = OmegaConf.load(config_path / "env.yaml")
        trials_config = OmegaConf.load(config_path / "trials.yaml")
        
        # Set experiment name from directory
        experiment_config.name = base_path.name
        
        return cls(
            name=base_path.name,
            base_path=base_path,
            experiment_config=experiment_config,
            model_config=model_config,
            env_config=env_config,
            trials_config=trials_config
        )

class ExperimentRunner:
    """Handles the lifecycle of running an experiment."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.experiment: Optional[Experiment] = None
    
    def setup(self) -> None:
        """
        Initialize the experiment and database.
        """
        # Create experiment instance
        self.experiment = Experiment.from_config(
            self.config.experiment_config,
            self.config.env_config
        )
        # Set model configuration
        self.experiment.config.settings = self.config.model_config
        
                    
    def run(self) -> None:
        """Run the experiment with all trials."""
        if self.experiment is None:
            raise RuntimeError("Experiment not initialized. Call setup() first.")
        
        try:
            self.experiment.run(self.config.trials_config)
        except Exception as e:
            raise RuntimeError(f"Failed to run experiment: {str(e)}")
    
    def run_full(self) -> None:
        """Run the complete experiment lifecycle."""
        self.setup()
        self.run()

def run_experiment(experiment_path: str) -> None:
    """Main entry point for running an experiment."""
    try:
        # Load all configurations
        config = ExperimentConfig.from_directory(experiment_path)
        
        # Create and run the experiment
        runner = ExperimentRunner(config)
        runner.run_full()
        
    except Exception as e:
        print(f"Error running experiment: {str(e)}")
        raise


SUBJECT = "simple"
EXPERIMENT_NAME = "simple"

def main():
    """
    runs the main logic
    """
    # Initialize the database
    DB.initialize("D:\\results\\DB\\experiment.db")
    
    # Construct the experiment path using the experiment name
    base_path = os.path.join("outputs", SUBJECT)
    experiment_path = os.path.join(base_path, EXPERIMENT_NAME)
    
    # Run the experiment
    run_experiment(experiment_path)

if __name__ == "__main__":
    main()
