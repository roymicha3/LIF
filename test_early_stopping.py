"""
Test script for early stopping functionality
"""
import os
import torch
from experiment_manager.experiment import Experiment
from pipeline.pipeline_factory import CustomPipelineFactory

EXPERIMENT_NAME = "test_early_stopping"
WORKSPACE = os.path.join("outputs", EXPERIMENT_NAME)

def main():
    """
    Test early stopping with a single trial
    """
    
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
    
    config_dir_path = os.path.join(
        os.path.dirname(__file__), 
        "configs", EXPERIMENT_NAME
        )
    
    print(f"Running early stopping test with config from: {config_dir_path}")
    print(f"Workspace: {WORKSPACE}")
    
    experiment = Experiment.create(config_dir_path, 
                                   factory=CustomPipelineFactory,
                                   workdir=WORKSPACE)
    
    print("Starting experiment...")
    experiment.run()
    print("Experiment completed!")

if __name__ == "__main__":
    main()
