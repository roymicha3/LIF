from experiment_manager.common.factory import Factory
from omegaconf import DictConfig
from torch.optim.lr_scheduler import (ConstantLR, CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts, CyclicLR,
                                      ExponentialLR, LinearLR, OneCycleLR,
                                      PolynomialLR, ReduceLROnPlateau, StepLR)


class LRSchedulerFactory(Factory):
    """
    Factory class for creating learning rate schedulers.
    """
    _registry = \
    {
        "StepLR": StepLR,
        "ConstantLR": ConstantLR,
        "LinearLR": LinearLR,
        "ExponentialLR": ExponentialLR,
        "PolynomialLR": PolynomialLR,
        "CyclicLR": CyclicLR,
        "ReduceLROnPlateau": ReduceLROnPlateau,
        "CosineAnnealingLR": CosineAnnealingLR, 
    }
    
    @staticmethod
    def create(name: str, optimizer, config: DictConfig):
        """
        Creates an instance of the specified learning rate scheduler.
        """
        if name not in LRSchedulerFactory._registry:
            raise ValueError(f"Scheduler '{name}' not found. Available: {list(LRSchedulerFactory._registry.keys())}")

        scheduler_class = LRSchedulerFactory._registry[name]

        try:
            if config.args is None:
                return scheduler_class(optimizer)
            
            return scheduler_class(optimizer, **config.args)  # Unpack DictConfig into keyword arguments
        except TypeError as e:
            raise ValueError(f"Invalid parameters for scheduler '{name}': {e}") from e
