from typing import Type, Dict
from omegaconf import DictConfig

from settings.factory import Factory

# import the kernel classes
from network.kernel.leaky_kernel import LeakyKernel
from network.kernel.den_kernel import DENKernel

class KernelFactory(Factory):
    """
    Factory class for creating kernels.
    """


