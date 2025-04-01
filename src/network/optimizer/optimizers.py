from omegaconf import DictConfig
from torch import optim

from settings.serializable import YAMLSerializable

@YAMLSerializable.register("Adam")
class Adam(optim.Adam, YAMLSerializable):
    """
    Adam optimizer without momentum.
    """

    def __init__(self, params, lr=0.01, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)

    @classmethod
    def from_config(cls, config: DictConfig, params):
        return cls(params,
               lr=config.get("lr", 0.01),
               betas=(config.get("beta_l", 0.9), config.get("beta_h", 0.999)),
               eps=config.get("eps", 1e-8),
               weight_decay=config.get("weight_decay", 0),
               amsgrad=config.get("amsgrad", False)) 