from omegaconf import DictConfig

from settings.factory import Factory
from settings.serializable import YAMLSerializable

# import the callbacks classes
from pipeline.callback.metric_tracker import MetricsTracker


class CallbackFactory(Factory):
    """
    Factory class for creating callbacks.
    """
    @staticmethod
    def create(name: str, config: DictConfig, env_config: DictConfig):
        """
        Create an instance of a registered callback.
        """
        return YAMLSerializable.get_by_name(name).from_config(config, env_config)

