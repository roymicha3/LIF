"""
the main of the project
"""
import os
from omegaconf import OmegaConf, DictConfig

from common import MODEL_NS, SPIKE_NS, DATA_NS
# from analysis.visualization import *
from experiment.trial import Trial
from network.kernel.kernel_factory import KernelFactory


def main():
    """
    runs the main logic
    """
    print("This is the thesis main!")
    
    # config = MODEL_ATTRIBUTES
    
    # Trial.run(config, report=False)
    
    config = OmegaConf.load("kernel.yaml")
    kernel = KernelFactory.create(config.type, config)
    
    # simple_tempotron_tune_hyperparameters()
    
    # visualizer = RandomSpikePattern(config)
    # visualizer.den_response()
    
    # RandomSpikePattern.results_b()
    
    

if __name__ == "__main__":
    main()
