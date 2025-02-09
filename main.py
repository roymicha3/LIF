"""
the main of the project
"""
import os
from omegaconf import OmegaConf, DictConfig


from experiment.experiment import Experiment


def main():
    """
    runs the main logic
    """
    print("This is the thesis main!")
    
    base_dir_path = os.path.join("outputs", "experiment example", "config")
    
    experiment_config = OmegaConf.load(os.path.join(base_dir_path, "experiment.yaml"))
    config = OmegaConf.load(os.path.join(base_dir_path, "config.yaml"))
    
    experiment_config.settings = config
    
    env_config = OmegaConf.load(os.path.join(base_dir_path, "env.yaml"))
    
    trials_config = OmegaConf.load(os.path.join(base_dir_path, "trials.yaml"))
    
    experiment = Experiment.from_config(experiment_config, env_config)
    
    experiment.run(trials_config)
    
    

if __name__ == "__main__":
    main()
