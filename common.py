"""
a module that provides common functionality for the model.
"""
from abc import ABC, abstractmethod
import yaml

class ComponentRegistry:
    """
    A static class that serves as a registry for layers, optimizers, loss functions...
    """

    REGISTRIES = {
        'layer': {},
        'optimizer': {},
        'loss': {}
    }

    @staticmethod
    def register(name, component_class, registry_type):
        """
        Register a component class in the specified registry.

        Args:
            name (str): The name of the component.
            component_class (type): The class of the component.
            registry_type (str): The type of the registry ('layer', 'optimizer', 'loss').
        """
        ComponentRegistry.REGISTRIES[registry_type][name] = component_class

    @staticmethod
    def get(registry_type):
        return ComponentRegistry.REGISTRIES.get(registry_type, {})


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