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
        # the configurations directory path to assemble the experiment
        self.base_dir = base_dir
        self.experiment_name = experiment_name
        self.config = config
        self.env_config = env_config

        self.experiment_dir = os.path.join(self.base_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)

        if not os.path.exists(self.base_dir):
            raise FileNotFoundError(f"Base directory '{self.base_dir}' does not exist.")

        self.id = Experiment.setup_experiment(self.experiment_name, config, env_config)

    @staticmethod
    def setup_experiment(experiment_name: str, experiment_conf: DictConfig, env_conf: DictConfig) -> int:
        """
        Setup the experiment directory and save the configuration.
        """
        
        # setup the database
        DB.initialize(experiment_conf.db_path)
        
        experiment_path = os.path.join(DB.instance().data_path, experiment_name)
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        
        # Save the experiment configuration
        experiment_conf_db_path = os.path.join(experiment_path, Experiment.CONFIG_FILE)
        
        OmegaConf.save(experiment_conf, experiment_conf_db_path)
        
        id_ = DB.instance().create_experiment(experiment_conf.name, experiment_conf.desc)
        
        # register the experiment configuration as artifact
        artifact_id = DB.instance().create_artifact("config", experiment_conf_db_path)
        DB.instance().add_artifact_to_experiment(id_, artifact_id)
        
        # Save the env configuration
        env_conf_db_path = os.path.join(experiment_path, "env.yaml")
        
        OmegaConf.save(env_conf, env_conf_db_path)
        
        # register the env configuration as artifact
        artifact_id = DB.instance().create_artifact("config", env_conf_db_path)
        DB.instance().add_artifact_to_experiment(id_, artifact_id)
        
        return id_

    def run(self, trial_config: DictConfig) -> None:
        """
        Run the experiment.
        """
        for conf in trial_config:
            conf.settings = OmegaConf.merge(self.config.settings, conf.settings)
            experiment_dir = os.path.join(DB.instance().data_path, self.experiment_name)
            if not os.path.exists(experiment_dir):
                os.makedirs(experiment_dir)
            
            conf.base_dir = experiment_dir
            trial = Trial.from_config(conf, self.env_config)
            trial.run(self.id)
            
    @classmethod
    def from_config(cls, config: DictConfig, env_config: DictConfig):
        return cls(config.name, 
                   config.base_dir,
                   config,
                   env_config)
