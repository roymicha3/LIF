from omegaconf import DictConfig
from torch.optim.lr_scheduler import StepLR, ConstantLR, LinearLR, ExponentialLR, PolynomialLR

from settings.factory import Factory

class LRSchedulerFactory(Factory):
    
    _registry = {
        "StepLR": StepLR,
        "ConstantLR": ConstantLR,
        "LinearLR": LinearLR,
        "ExponentialLR": ExponentialLR,
        "PolynomialLR": PolynomialLR
    }
    
    @staticmethod
    def create(name: str, optimizer, config: DictConfig):
        """
        Creates an instance of the specified learning rate scheduler.

        Args:
            name (str): Name of the scheduler to create.
            optimizer (torch.optim.Optimizer): Optimizer instance to attach the scheduler to.
            config (DictConfig): Configuration parameters for the scheduler.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: Instantiated scheduler.

        Raises:
            ValueError: If the scheduler name is not found or invalid parameters are provided.
        """
        if name not in LRSchedulerFactory._registry:
            raise ValueError(f"Scheduler '{name}' not found. Available: {list(LRSchedulerFactory._registry.keys())}")

        scheduler_class = LRSchedulerFactory._registry[name]

        try:
            return scheduler_class(optimizer, **config.args)  # Unpack DictConfig into keyword arguments
        except TypeError as e:
            raise ValueError(f"Invalid parameters for scheduler '{name}': {e}") from e
