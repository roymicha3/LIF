from omegaconf import DictConfig

from settings.factory import Factory
from settings.serializable import YAMLSerializable

# Import all the loss
from network.loss.binary_loss import BinaryLoss

class LossFactory(Factory):
    """
    Factory class for creating losses.
    """
    @staticmethod
    def create(name: str, config: DictConfig, env_config: DictConfig):
        """
        Create an instance of a registered loss.
        """
        class_ = YAMLSerializable.get_by_name(name)
        return class_.from_config(config, env_config)
