"""
the main of the project
"""
from common import init_model_attributes, MODEL_NS, SPIKE_NS
from analysis.visualization import *


MODEL_ATTRIBUTES = \
{
    # MODEL PARAMETERS:
    MODEL_NS.BATCH_SIZE : 1,
    MODEL_NS.DATASET_SIZE : 640,
    MODEL_NS.NUM_CLASSES : 2,
    MODEL_NS.NUM_OUTPUTS : 1,
    MODEL_NS.NUM_INPUTS : 500,
    MODEL_NS.TRAINING_PERCENTAGE : 50,
    MODEL_NS.TESTING_PERCENTAGE : 25,
    MODEL_NS.VALIDATION_PERCENTAGE : 25,
    
    # SPIKE PARAMETERS:
    SPIKE_NS.T : 500,
    SPIKE_NS.dt : 1.0,
    SPIKE_NS.tau: 10,
    
    SPIKE_NS.tau_m: 15,
    SPIKE_NS.tau_s: 15 / 4,
    SPIKE_NS.v_thr: 1,
}

def main():
    """
    runs the main logic
    """
    print("This is the thesis main!")
    
    init_model_attributes(MODEL_ATTRIBUTES)
    
    visualizer = RandomSpikePattern(ATTR(MODEL_NS.BATCH_SIZE))
    
    visualizer.train_max_time()
    

if __name__ == "__main__":
    main()
