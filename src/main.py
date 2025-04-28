"""
the main of the project
"""
import os

from pipeline.pipeline_factory import PipelineFactory

from experiment_manager.experiment import Experiment



SUBJECT = "simple"
EXPERIMENT_NAME = "simple"

def main():
    """
    runs the main logic
    """
    
    config_dir_path = os.path.join(
        os.path.dirname(__file__), 
        os.path.pardir, 
        "configs", EXPERIMENT_NAME
        )
    
    experiment = Experiment.create(config_dir_path, factory = PipelineFactory)
    experiment.run()
    
if __name__ == "__main__":
    main()
