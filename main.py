"""
the main of the project
"""
import os

from common import MODEL_NS, SPIKE_NS, DATA_NS, Configuration
from analysis.visualization import *
from experiment.trial import Trial
from experiment.experiment import simple_tempotron_tune_hyperparameters

from analysis.results import RandomSpikePattern


MODEL_ATTRIBUTES = \
{
    # MODEL PARAMETERS:
    MODEL_NS.NUM_OUTPUTS             : 1,
    MODEL_NS.NUM_INPUTS              : 500,
    MODEL_NS.LR                      : 100.0
    ,
    MODEL_NS.MOMENTUM                : 0.99,
    MODEL_NS.EPOCHS                  : 1000,
    MODEL_NS.BETA                    : 50,
    
    # DATA PARAMETERS:
    DATA_NS.BATCH_SIZE               : 64,
    DATA_NS.DATASET_SIZE             : 1000,
    DATA_NS.NUM_CLASSES              : 2,
    DATA_NS.TRAINING_PERCENTAGE      : 50,
    DATA_NS.TESTING_PERCENTAGE       : 25,
    DATA_NS.VALIDATION_PERCENTAGE    : 100,
    DATA_NS.ROOT                     : os.path.join(".", "data", "data", "random"),
    
    # SPIKE PARAMETERS:
    SPIKE_NS.T                       : 500,
    SPIKE_NS.dt                      : 1.0,
    SPIKE_NS.tau                     : 10,
    
    SPIKE_NS.tau_m                   : 10,
    SPIKE_NS.tau_s                   : 10 / 4,
    SPIKE_NS.v_thr                   : 1,
}

def main():
    """
    runs the main logic
    """
    # print("This is the thesis main!")
    
    config = Configuration(MODEL_ATTRIBUTES)
    
    Trial.run(config, report=False)
    
    # simple_tempotron_tune_hyperparameters()
    
    # visualizer = RandomSpikePattern(config)
    # visualizer.den_response()
    
    # RandomSpikePattern.results_b()
    
    

if __name__ == "__main__":
    main()
