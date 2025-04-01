from settings.factory import Factory
from omegaconf import DictConfig

from settings.serializable import YAMLSerializable

# import the kernel classes
from network.kernel.leaky_kernel import LeakyKernel
from network.kernel.den_kernel import DENKernel

class KernelFactory(Factory):
    """
    Factory class for creating kernels.
    """
    @staticmethod
    def create(name: str, config: DictConfig, env_config: DictConfig):
        """
        Create an instance of a registered kernel.
        """
        return YAMLSerializable.get_by_name(name).from_config(config, env_config)

