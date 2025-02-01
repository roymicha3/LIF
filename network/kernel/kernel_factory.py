from typing import Type, Dict

class KernelFactory:
    """
    Factory class for creating kernels.
    """
    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a model class with a given name.
        """
        def decorator(kernel_class: Type):
            cls._registry[name] = kernel_class
            return kernel_class
        return decorator

    @classmethod
    def create(cls, name: str, config: dict):
        """
        Create an instance of a registered model.
        """
        if name not in cls._registry:
            raise ValueError(f"Kernel '{name}' is not registered.")
        return cls._registry[name](config)