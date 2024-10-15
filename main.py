"""
the main of the project
"""
import os

from common import MODEL_NS, SPIKE_NS, DATA_NS, Configuration
from analysis.visualization import *


MODEL_ATTRIBUTES = \
{
    # MODEL PARAMETERS:
    MODEL_NS.NUM_OUTPUTS             : 1,
    MODEL_NS.NUM_INPUTS              : 500,
    MODEL_NS.LR                      : 0.01,
    
    # DATA PARAMETERS:
    DATA_NS.BATCH_SIZE               : 1,
    DATA_NS.DATASET_SIZE             : 640,
    DATA_NS.NUM_CLASSES              : 2,
    DATA_NS.TRAINING_PERCENTAGE      : 50,
    DATA_NS.TESTING_PERCENTAGE       : 25,
    DATA_NS.VALIDATION_PERCENTAGE    : 25,
    DATA_NS.ROOT                     : os.path.join("D:", "data", "random"),
    
    # SPIKE PARAMETERS:
    SPIKE_NS.T                       : 500,
    SPIKE_NS.dt                      : 1.0,
    SPIKE_NS.tau: 10,
    
    SPIKE_NS.tau_m                   : 15,
    SPIKE_NS.tau_s                   : 15 / 4,
    SPIKE_NS.v_thr                   : 1,
}

def main():
    """
    runs the main logic
    """
    print("This is the thesis main!")
    
    config = Configuration(MODEL_ATTRIBUTES)
    
    visualizer = RandomSpikePattern(config)
    
    # os.environ['RAY_TMPDIR'] = "D:\\"
    
    visualizer.train_max_time()
    

if __name__ == "__main__":
    main()
