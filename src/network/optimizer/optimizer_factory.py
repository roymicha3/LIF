from omegaconf import DictConfig

from experiment_manager.common.factory import Factory
from experiment_manager.common.serializable import YAMLSerializable

# Import all the optimizers
from network.optimizer.momentum_opt import MomentumOptimizer
from network.optimizer.optimizers import *

class OptimizerFactory(Factory):
    """
    Factory class for creating optimizers.
    """

    @staticmethod
    def create(name: str, config: DictConfig, params):
        """
        Create an instance of a registered optimizer.
        """
        class_ = YAMLSerializable.get_by_name(name)
        return class_.from_config(config, params)