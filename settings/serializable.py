from omegaconf import OmegaConf, DictConfig


class YAMLSerializable:
    """
    Base class for YAML serialization support.
    """
    
    def __init__(self, config: DictConfig = None):
        self.config = config

    def save(self, file_path):
        """
        Save model architecture to YAML using OmegaConf.
        """
        pass

    @classmethod
    def load(cls, file_path):
        """
        Load model architecture from YAML.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            config = OmegaConf.load(f)
        
        return cls.from_config(config)

    @classmethod
    def from_config(cls, config: DictConfig):
        """
        Create an instance from a DictConfig.
        """
        return cls(config)