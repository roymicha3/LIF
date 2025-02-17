import os
from omegaconf import OmegaConf, DictConfig

# TODO: someday support multiple kinds of pipelines...(consider creating a factory for pipelines)
from settings.serializable import YAMLSerializable
from pipeline.training_pipeline import TrainingPipeline

class Trial(YAMLSerializable):
    """
    Trial class is the trial in the experiment.
    """
    TRIAL_CONFIG = "trial_config.yaml"

    def __init__(self, 
                 name: str,
                 base_dir: str,
                 repeat: int,
                 config: DictConfig,
                 env_conf: DictConfig):
        
        super().__init__()
        self.base_dir = base_dir
        self.trial_name = name
        self.repeat = repeat

        self.trial_dir = os.path.join(self.base_dir, self.trial_name)

        if not os.path.exists(self.base_dir):
            raise FileNotFoundError(f"Base directory '{self.base_dir}' does not exist.")

        self.env_conf = env_conf
        self.trial_conf = config
        Trial.setup_trial(self.trial_dir, config)

    @staticmethod
    def setup_trial(trial_dir: str, config: DictConfig) -> DictConfig:
        """
        Setup the trial directory and save the configuration.
        """
        os.makedirs(trial_dir, exist_ok=True)

        # Save the trial configuration
        trial_config_path = os.path.join(trial_dir, Trial.TRIAL_CONFIG)
        OmegaConf.save(config, trial_config_path)
        

    def run_single(self, tag: str) -> None:
        """
        Run a single trial.
        """
        repeat_path = os.path.join(self.trial_dir, tag)
        os.makedirs(repeat_path, exist_ok=True)
        
        # set the working directory
        env_conf = self.env_conf.copy()
        env_conf.work_dir = repeat_path
        
        # Create pipeline
        pipeline = TrainingPipeline.from_config(self.trial_conf.pipeline, env_conf)
        pipeline.run(self.trial_conf, env_conf)

    def run(self) -> None:
        """
        Run the trial.
        """
        for i in range(self.repeat):
            self.run_single(f"repeat_{i}")
            
    @classmethod
    def from_config(cls, config, env_config):
        return cls(config.name, config.base_dir, config.repeat, config.settings, env_config)