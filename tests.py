from omegaconf import DictConfig, OmegaConf

from network.kernel.kernel_factory import KernelFactory


def test_kernel_factory():
    config = OmegaConf.load("kernel.yaml")
    env_config = OmegaConf.load("env.yaml")
    
    kernel = KernelFactory.create(config.type, config, env_config)
    assert kernel is not None
    

test_kernel_factory()
    
