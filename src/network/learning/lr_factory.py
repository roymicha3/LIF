from experiment_manager import Environment
from experiment_manager.common import Factory, YAMLSerializable
from omegaconf import DictConfig

from network.learning.integrate_lr import IntegrateLearningRule
from network.learning.sequential.plasticity_induction import \
    PlasticityInduction
from network.learning.sequential.single_spike_lr import SequentialSingleSpikeLR
# Import all the learning rules
from network.learning.single_spike_lr import SingleSpikeLR


class LearningRuleFactory(Factory):
    """
    Factory class for creating learning rules.
    """
    
    @staticmethod
    def create(name, config: DictConfig, env: Environment):
        return YAMLSerializable.get_by_name(name).from_config(config, env)
