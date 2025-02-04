from omegaconf import DictConfig

from settings.factory import Factory
from settings.serializable import YAMLSerializable

# import all encoders:
from encoders.identity_encoder import IdentityEncoder
from encoders.spike.latency_encoder import LatencyEncoder
from encoders.spike.rate_encoder import RateEncoder


class EncoderFactory(Factory):
    """
    Factory class for creating kernels.
    """
    @staticmethod
    def create(name: str, config: DictConfig, env_config: DictConfig):
        """
        Create an instance of a registered kernel.
        """
        return YAMLSerializable.get_by_name(name).from_config(config, env_config)