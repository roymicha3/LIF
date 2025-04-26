"""
the main of the project
"""
import os
from experiment_manager.experiment import Experiment



SUBJECT = "simple"
EXPERIMENT_NAME = "simple"

def main():
    """
    runs the main logic
    """
    
    config_dir_path = os.path.join(os.path.dirname(__file__), "configs", EXPERIMENT_NAME)
    experiment = Experiment.create(config_dir_path, factory = None)
    # experiment.run()
    
if __name__ == "__main__":
    main()
