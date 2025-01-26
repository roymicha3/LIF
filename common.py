"""
a module that provides common functionality for the model.
"""
from abc import ABC, abstractmethod
import yaml

from settings.spike.spike_namespace import SpikeNamespace
from settings.model_namespace import ModelNamespace
from settings.data_namespace import DataNamespace

from settings.config import Configuration

SPIKE_NS = SpikeNamespace
MODEL_NS = ModelNamespace
DATA_NS = DataNamespace


class YamlSerializable(ABC):
    """
    Base class for components that can be serialized to and deserialized from YAML.
    """

    @abstractmethod
    def to_yaml(self):
        """
        Serialize the component to a YAML-compatible dictionary.

        Returns:
            dict: A dictionary representing the component's configuration.
        """
        pass

    @classmethod
    @abstractmethod
    def from_yaml(cls, config):
        """
        Deserialize the component from a YAML-compatible dictionary.

        Args:
            config (dict): A dictionary containing the component's configuration.

        Returns:
            YamlSerializable: An instance of the component.
        """
        pass