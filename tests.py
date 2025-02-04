from omegaconf import DictConfig, OmegaConf

from network.network_factory import NetworkFactory


def test_network_factory():
    config = OmegaConf.load("config.yaml")
    env_config = OmegaConf.load("env.yaml")
    
    model_config = config.model
    network = NetworkFactory.create(model_config.type, model_config, env_config)
    assert network is not None
    

test_network_factory()
    
