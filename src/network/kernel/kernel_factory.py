from omegaconf import DictConfig

from experiment_manager.common.factory import Factory
from experiment_manager.environment import Environment
from experiment_manager.common.serializable import YAMLSerializable

# import the kernel classes
from network.kernel.den_kernel import DENKernel
from network.kernel.leaky_kernel import LeakyKernel
from network.kernel.sequential_kernel import SequentialLeakyKernel

class KernelFactory(Factory):
    """
    Factory class for creating kernels.
    """
    @staticmethod
    def create(name: str, config: DictConfig, env: Environment) -> YAMLSerializable:
        """
        Create an instance of a registered kernel.
        """
        return YAMLSerializable.get_by_name(name).from_config(config, env)

