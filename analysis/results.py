"""
Here we visualize our data and results
"""
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from ray import tune

from encoders.spike.latency_encoder import LatencyEncoder
from data.dataset.random_dataset import RandomDataset, DataType, OutputType
from common import Configuration, SPIKE_NS, MODEL_NS, DATA_NS

from network.nodes.node import Node
from network.nodes.leaky_node import LeakyNode
from network.nodes.den_node import DENNode
from network.nodes.single_spike_node import SingleSpikeNode
from network.topology.connection import Connection
from network.topology.network import Network
from network.learning.optimizers import MomentumOptimizer

from network.loss.binary_loss import BinaryLoss

class RandomSpikePattern:
    """
    visualize the random spike pattern data
    """
    
############################## Config: ############################## 
    
    MODEL_ATTRIBUTES = \
    {
        # MODEL PARAMETERS:
        MODEL_NS.NUM_OUTPUTS             : 1,
        MODEL_NS.NUM_INPUTS              : 500,
        MODEL_NS.LR                      : 1.0
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


        
    dataset = RandomDataset(
        MODEL_ATTRIBUTES,
        DataType.TRAIN,
        OutputType.TORCH, 
        LatencyEncoder(MODEL_ATTRIBUTES, 1))
    

############################## Results functions: ############################## 
    
    @staticmethod
    def results_a():
        """
        this section visualizes the spike generated data for the simple Tempotron
        """
        IDX = 0
        ENCODED = True
        
        data = RandomSpikePattern.dataset.get_raw(idx=IDX, encoded=ENCODED)
        data.plot()
        
    
    @staticmethod
    def results_b():
        """
        this section visualizes the DEN response to a randomly generated spikes
        """
        IDX = 0
        data, label = RandomSpikePattern.dataset[IDX]
        raw_data = RandomSpikePattern.dataset.get_raw(IDX)
        
        seq_len, n = data.shape
        
        kernel = DENNode(RandomSpikePattern.MODEL_ATTRIBUTES, n, scale=True)
        
        with torch.no_grad():
            response = kernel.forward(data)

        simulated_response = DENNode.assimulate_response(
            raw_data, 
            RandomSpikePattern.MODEL_ATTRIBUTES[SPIKE_NS.tau_m], 
            RandomSpikePattern.MODEL_ATTRIBUTES[SPIKE_NS.tau_s], 
            RandomSpikePattern.MODEL_ATTRIBUTES[SPIKE_NS.dt])
        
        simulated_response = simulated_response.numpy()
        
        # Plot the results
        plt.figure()

        plt.plot(response[:, 0], label='Output Spikes', color='red')
        plt.plot(simulated_response[:, 0], label='Simulated Output Spikes', color='blue', linestyle=":")
        plt.title('Model Output Spikes')
        plt.xlabel('Time Steps')
        plt.ylabel('Spike')
        plt.legend()

        plt.tight_layout()
        plt.show()