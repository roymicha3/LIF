from abc import ABC, abstractmethod
from enum import Enum

class Metrics(Enum):
    EPOCH       = "epoch"
    TEST_ACC    = "test_acc"
    TEST_LOSS   = "test_loss"
    VAL_ACC     = "val_acc"
    VAL_LOSS    = "val_loss"
    

class Callback(ABC):
    @abstractmethod
    def on_epoch_end(self, metrics) -> bool:
        """Called at the end of each epoch."""
        pass

    @abstractmethod
    def on_train_end(self, metrics):
        """Called at the end of training."""
        pass
