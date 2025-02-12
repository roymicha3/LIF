from omegaconf import DictConfig

from pipeline.callback.callback import Callback, Metrics
from settings.serializable import YAMLSerializable


@YAMLSerializable.register("EarlyStopping")
class EarlyStopping(Callback):
    """
    Early stops the training if validation metric doesn't improve after a given patience.
    """
    def __init__(self, metric=Metrics.VAL_LOSS, patience=5, min_delta_percent=0.0, verbose=False):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
            min_delta_percent (float): Minimum percentage change to qualify as an improvement.
            verbose (bool): Print message when stopping early.
        """
        self.metric = metric
        self.patience = patience
        self.min_delta_percent = min_delta_percent
        self.verbose = verbose
        self.counter = 0
        self.best_metric = None
        self.early_stop = False

    def on_epoch_end(self, metrics) -> bool:
        """
        Check if validation metric has improved; otherwise increase counter.

        Args:
            metrics (dict): Current metrics including the monitored metric.
        """
        current_metric = metrics[self.metric]
        
        if self.best_metric is None:
            self.best_metric = current_metric
            return False

        percent_change = (self.best_metric - current_metric) / self.best_metric * 100

        if percent_change > self.min_delta_percent:
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered")
                    
        return self.early_stop
    
    def on_train_end(self, metrics):
        pass
    
    @classmethod
    def from_config(cls, config: DictConfig, env_config: DictConfig):
        return cls(config.metric, config.patience, config.min_delta, config.verbose)
