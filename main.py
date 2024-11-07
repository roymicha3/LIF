import os
import torch
import torch.utils.data.dataset

from encoders.spike.latency_encoder import LatencyEncoder
from data.dataset.random_dataset import RandomDataset, DataType, OutputType
from network.kernel.den_kernel import DENKernel
from network.topology.simple_connection import SimpleConnection
from common import Configuration, SPIKE_NS, MODEL_NS, DATA_NS

from network.neuron.single_spike_neuron import SingleSpikeNeuron, NeuronOutputType
from experiment.trial import Trial


MODEL_ATTRIBUTES = \
{
    # MODEL PARAMETERS:
    MODEL_NS.NUM_OUTPUTS             : 1,
    MODEL_NS.NUM_INPUTS              : 500,
    MODEL_NS.LR                      : 0.01,
    MODEL_NS.MOMENTUM                : 0.99,
    MODEL_NS.EPOCHS                  : 1000,
    MODEL_NS.BETA                    : 50,
    
    # DATA PARAMETERS:
    DATA_NS.BATCH_SIZE               : 1,
    DATA_NS.DATASET_SIZE             : 1000,
    DATA_NS.NUM_CLASSES              : 2,
    DATA_NS.TRAINING_PERCENTAGE      : 50,
    DATA_NS.TESTING_PERCENTAGE       : 25,
    DATA_NS.VALIDATION_PERCENTAGE    : 50,
    DATA_NS.ROOT                     : os.path.join(".", "data", "data", "random"),
    
    # SPIKE PARAMETERS:
    SPIKE_NS.T                       : 500,
    SPIKE_NS.dt                      : 1.0,
    SPIKE_NS.tau                     : 10,
    
    SPIKE_NS.tau_m                   : 15,
    SPIKE_NS.tau_s                   : 15 / 4,
    SPIKE_NS.v_thr                   : 1,
}

SUCCESS = 0
ERROR = 1

def main():
    print("Playing with pytorch :)")
    
    
    # Set device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    Trial.run(MODEL_ATTRIBUTES,False, early_stopping_patience = 25)
    
    # # Initialize dataset and move it to the correct device
    # dataset = RandomDataset(
    #     MODEL_ATTRIBUTES,
    #     DataType.TRAIN,
    #     OutputType.NORMAL,
    #     LatencyEncoder(MODEL_ATTRIBUTES, 1))
    
    # kernel = DENKernel(MODEL_ATTRIBUTES, MODEL_ATTRIBUTES[MODEL_NS.NUM_INPUTS], device)
    # connection = SimpleConnection(
    #     dim=(MODEL_ATTRIBUTES[MODEL_NS.NUM_INPUTS], MODEL_ATTRIBUTES[MODEL_NS.NUM_OUTPUTS]),
    #     device=device)
    
    # neuron = SingleSpikeNeuron(MODEL_ATTRIBUTES, kernel, connection, type_=NeuronOutputType.VALUE)
    
    # IDX = 5
    
    # data, label = dataset[IDX]
    
    # res = neuron.forward(data)
    
    
    # neuron.plot_voltages(data)
    
    return SUCCESS

if __name__ == '__main__':
    main()
