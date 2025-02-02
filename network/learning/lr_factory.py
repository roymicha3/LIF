from omegaconf import DictConfig

from settings.factory import Factory
from settings.serializable import YAMLSerializable

# Import all the learning rules
from network.learning.single_spike_lr import SingleSpikeLR

class LearningRuleFactory(Factory):
    """
    Factory class for creating learning rules.
    """
