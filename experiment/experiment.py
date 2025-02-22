import os
from omegaconf import OmegaConf, DictConfig

from experiment.db.database import DB

from experiment.trial import Trial
from settings.serializable import YAMLSerializable

class Experiment(YAMLSerializable):
    
    CONFIG_FILE = "experiment_config.yaml"
    
    def __init__(self, 
                 experiment_name: str, 
                 base_dir: str,
                 config: DictConfig,
                 env_config: DictConfig):
        
        super().__init__()
        self.base_dir = base_dir
        self.experiment_name = experiment_name
        self.config = config
        self.env_config = env_config

        self.experiment_dir = os.path.join(self.base_dir, self.experiment_name)

        if not os.path.exists(self.base_dir):
            raise FileNotFoundError(f"Base directory '{self.base_dir}' does not exist.")

        self.id = Experiment.setup_experiment(self.experiment_dir, config)

    @staticmethod
    def setup_experiment(experiment_dir: str, experiment_conf: DictConfig):
        """
        Setup the experiment directory and save the configuration.
        """
        os.makedirs(experiment_dir, exist_ok=True)

        # Save the experiment configuration
        experiment_config_path = os.path.join(experiment_dir, Experiment.CONFIG_FILE)
        OmegaConf.save(experiment_conf, experiment_config_path)
        
        # setup the database
        db_path = os.path.join(experiment_dir, "db")
        DB.initialize(db_path)
        
        db_data_path = DB.instance().db_dir_path
        experiment_conf_db_path = os.path.join(db_data_path, Experiment.CONFIG_FILE)
        OmegaConf.save(experiment_conf, experiment_conf_db_path)
        
        id_ = DB.instance().create_experiment(experiment_conf.name, experiment_conf.desc)
        return id_

    def run(self, trial_config: DictConfig) -> None:
        """
        Run the experiment.
        """
        for conf in trial_config:
            conf.settings = OmegaConf.merge(self.config.settings, conf.settings)
            conf.base_dir = self.experiment_dir
            trial = Trial.from_config(conf, self.env_config)
            trial.run()
            
    @classmethod
    def from_config(cls, config: DictConfig, env_config: DictConfig):
        return cls(config.name, 
                   config.base_dir,
                   config,
                   env_config)
