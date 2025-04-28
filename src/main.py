"""
the main of the project
"""
import os
import torch

from pipeline.pipeline_factory import PipelineFactory

from experiment_manager.experiment import Experiment


EXPERIMENT_NAME = "length"
WORKSPACE = os.path.join("outputs", "length_workspace")

def main():
    """
    runs the main logic
    """
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your installation.")
    
    config_dir_path = os.path.join(
        os.path.dirname(__file__), 
        os.path.pardir, 
        "configs", EXPERIMENT_NAME
        )
    
    experiment = Experiment.create(config_dir_path, 
                                   factory = PipelineFactory,
                                   workdir=WORKSPACE)
    experiment.run()
    
if __name__ == "__main__":
    main()
