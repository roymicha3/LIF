from omegaconf import DictConfig

from settings.factory import Factory
from settings.serializable import YAMLSerializable

# Import all the loss
from network.loss.binary_loss import BinaryLoss

class LossFactory(Factory):
    """
    Factory class for creating losses.
    """
