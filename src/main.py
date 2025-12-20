"""
the main of the project
"""
import os

import torch
from experiment_manager.experiment import Experiment
from experiment_manager.common.factory_registry import FactoryRegistry, FactoryType

from pipeline.pipeline_factory import CustomPipelineFactory
from pipeline.callbacks.callback_factory import CustomCallbackFactory

EXPERIMENT_NAME = "optimal_experiment"

# Workspace at project root level (not inside src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE = os.path.join(PROJECT_ROOT, "outputs", EXPERIMENT_NAME)


def main():
    """
    runs the main logic
    """
    
    config_dir_path = os.path.join(
        os.path.dirname(__file__), 
        os.path.pardir, 
        "configs", EXPERIMENT_NAME
        )
    
    # Create factory registry with custom factories
    registry = FactoryRegistry()
    registry.register(FactoryType.PIPELINE, CustomPipelineFactory())
    registry.register(FactoryType.CALLBACK, CustomCallbackFactory())
    
    experiment = Experiment.create(
        config_dir_path, 
        factory_registry=registry,
        workdir=WORKSPACE
    )
    experiment.run()


if __name__ == "__main__":
    main()
