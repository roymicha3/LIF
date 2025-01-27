"""
the main of the project
"""
import os

from common import *
from analysis.visualization import *
from experiment.trial import Trial
from experiment.experiment import simple_tempotron_tune_hyperparameters

from analysis.results import RandomSpikePattern



def main():
    """
    runs the main logic
    """
    print("This is the thesis main!")
    
    Trial.run(config, report=False)
    
    # simple_tempotron_tune_hyperparameters()
    
    # visualizer = RandomSpikePattern(config)
    # visualizer.den_response()
    
    # RandomSpikePattern.results_b()
    
    

if __name__ == "__main__":
    main()
