"""
Custom callback factory that registers project-specific callbacks.

This factory inherits from CallbackFactory to leverage the standard
callback creation logic while allowing registration of custom callbacks.
"""
from omegaconf import DictConfig

from experiment_manager import Environment
from experiment_manager.pipelines import CallbackFactory

# Import to register via @YAMLSerializable.register decorator
from pipeline.callbacks.visualization_callback import VisualizationCallback


class CustomCallbackFactory(CallbackFactory):
    """
    Factory for creating project-specific callbacks.
    
    Inherits from CallbackFactory to use the standard creation logic.
    Custom callbacks are registered via @YAMLSerializable.register decorator
    and automatically available through this factory.
    """
    
    @staticmethod
    def create(name: str, config: DictConfig, env: Environment):
        """
        Create a callback instance from configuration.
        
        Args:
            name: The registered name of the callback (e.g., "VisualizationCallback")
            config: Configuration dict for the callback
            env: Environment instance
            
        Returns:
            Callback instance
        """
        # Delegate to parent factory - all callbacks (built-in and custom)
        # are available via YAMLSerializable registry
        return CallbackFactory.create(name, config, env)

