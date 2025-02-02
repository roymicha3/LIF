from omegaconf import DictConfig, OmegaConf

from network.kernel.kernel_factory import KernelFactory


def test_kernel_factory():
    config = OmegaConf.load("kernel.yaml")
    config = OmegaConf.merge(config.kernel, config.env)
    kernel = KernelFactory.create(config.type, config)
    assert kernel is not None
    

test_kernel_factory()
    
