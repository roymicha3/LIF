from omegaconf import DictConfig

from settings.factory import Factory
from settings.serializable import YAMLSerializable

# import all the spike encoders
from encoders.spike.latency_encoder import LatencyEncoder
from encoders.spike.rate_encoder import RateEncoder

class SpikeEncoderFactory(Factory):
    """
    Factory class for creating spike encoders.
    """
    @staticmethod
    def create(name: str, config: DictConfig, env_config: DictConfig):
        """
        Create an instance of a registered spike encoder.
        """
        return YAMLSerializable.get_by_name(name).from_config(config, env_config)