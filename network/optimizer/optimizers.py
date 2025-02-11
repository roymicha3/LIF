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
                   lr=config.lr,
                   betas=(config.beta_l, config.beta_h),
                   eps=config.eps,
                   weight_decay=config.weight_decay,
                   amsgrad=config.amsgrad) 