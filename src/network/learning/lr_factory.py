from omegaconf import DictConfig

from experiment_manager.common.factory import Factory
from experiment_manager.environment import Environment
from experiment_manager.common.serializable import YAMLSerializable

# Import all the learning rules
from network.learning.single_spike_lr import SingleSpikeLR
from network.learning.integrate_lr import IntegrateLearningRule

class LearningRuleFactory(Factory):
    """
    Factory class for creating learning rules.
    """
    
    @staticmethod
    def create(name, config: DictConfig, env: Environment):
        return YAMLSerializable.get_by_name(name).from_config(config, env)
