import os
from omegaconf import DictConfig, OmegaConf
import torch

from network.network_factory import NetworkFactory, Network
from pipeline.training_pipeline import TrainingPipeline
from pipeline.plotting_pipeline import PlottingPipeline


def test_network_factory():
    config = OmegaConf.load("config.yaml")
    env_config = OmegaConf.load("env.yaml")
    
    model_config = config.model
    network = NetworkFactory.create(model_config.type, model_config, env_config)
    assert network is not None
    

def test_training_pipeline():
    config = OmegaConf.load("config.yaml")
    env_config = OmegaConf.load("env.yaml")
    
    pipeline = TrainingPipeline.from_config(config.pipeline, env_config)
    
    pipeline.run(config, env_config)
    

base_dir = os.path.join("outputs", "single run")
config = OmegaConf.load(os.path.join(base_dir, "config", "config.yaml"))
env_config = OmegaConf.load(os.path.join(base_dir, "config", "env.yaml"))

pipeline = PlottingPipeline(base_dir)
pipeline.run(config, env_config)
