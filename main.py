"""
the main of the project
"""
import os
from omegaconf import OmegaConf

from experiment.experiment import Experiment


def run_experiment(experiment_path: str):
    
    if not os.path.exists(experiment_path):
        print(f"Base directory path '{experiment_path}' does not exist.")
        raise NotADirectoryError(f"Base directory path '{experiment_path}' does not exist.")
    
    config_path = os.path.join(experiment_path, "config")
    if not os.path.exists(config_path):
        print(f"Config path '{config_path}' does not exist.")
        raise NotADirectoryError(f"Config path '{config_path}' does not exist.")
    
    # build experiment configuration:
    experiment_config = OmegaConf.load(os.path.join(config_path, "experiment.yaml"))
    
    # set the epxeriment name according to the directory
    experiment_config.name = os.path.basename(experiment_path)
    
    # base config of the architecture
    config = OmegaConf.load(os.path.join(config_path, "config.yaml"))
    experiment_config.settings = config
    
    # environment config of the experiment
    env_config = OmegaConf.load(os.path.join(config_path, "env.yaml"))
    
    # the trial entries:
    trials_config = OmegaConf.load(os.path.join(config_path, "trials.yaml"))
    
    experiment = Experiment.from_config(experiment_config, env_config)
    
    experiment.run(trials_config)


EXPERIMENT_NAME = "single run" # "integral lr beta validation"

def main():
    """
    runs the main logic
    """
    print("This is the thesis main!")
    base_dir_path = os.path.join("outputs", EXPERIMENT_NAME)
    
    run_experiment(base_dir_path)


if __name__ == "__main__":
    main()
