from omegaconf import DictConfig, OmegaConf

from network.network_factory import NetworkFactory
from pipeline.training_pipeline import TrainingPipeline


def test_network_factory():
    config = OmegaConf.load("config.yaml")
    env_config = OmegaConf.load("env.yaml")
    
    model_config = config.model
    network = NetworkFactory.create(model_config.type, model_config, env_config)
    assert network is not None
    

def test_training_pipeline():
    config = OmegaConf.load("config.yaml")
    env_config = OmegaConf.load("env.yaml")
    
    pipeline = TrainingPipeline.from_config(config.pipeline)
    
    pipeline.run(config, env_config)
    

test_training_pipeline()