import os
import torch
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf

from pipeline.pipline import Pipeline
from settings.serializable import YAMLSerializable

from tools.utils import get_prefix_files


class PlottingPipeline(Pipeline, YAMLSerializable):
    """
    class that is responsible for the Plotting of the Network
    """
    def __init__(self, work_dir: str):
        super(PlottingPipeline, self).__init__()
        super(YAMLSerializable, self).__init__()
        
        self.work_dir = work_dir
    
    
    @classmethod
    def from_config(cls, config: DictConfig, env_config: DictConfig): # TODO: fix this
        pipeline = cls()
            
        return pipeline
    
    
    def run(self, config: DictConfig, env_config: DictConfig):
        """
        Plot the model using the provided data loader.
        """
        dataset = self.load_dataset(config.dataset, env_config)
        
        sample = dataset.get_raw(0, encoded=True)
        sample.plot()
        
        checkpoints = get_prefix_files(self.work_dir, "checkpoint")
        
        for checkpoint in checkpoints:
            
            with torch.no_grad():
                network = torch.load(checkpoint)
                response = network.inner_state(sample, -1)
            
            # Plot the results
            plt.figure()

            plt.plot(response[:, 0].to("cpu"), label='Output Spikes', color='red')
            plt.title('Model Output Spikes')
            plt.xlabel('Time Steps')
            plt.ylabel('Spike')
            plt.legend()

            plt.tight_layout()
            plt.savefig(
                os.path.join(os.path.dirname(checkpoint),
                             f"voltage_{os.path.basename(checkpoint)}.png"))
            plt.close()
            
    
    def evaluate(self, network, criterion, dataset):
        pass
        

    def save(self, file_path):
        """
        Save the training pipeline configuration to YAML.
        """
        config = DictConfig({ })
        with open(file_path, 'w', encoding='utf-8') as f:
            OmegaConf.save(config, f)
