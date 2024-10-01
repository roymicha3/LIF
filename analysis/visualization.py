"""
Here we visualize our data and results
"""
import matplotlib.pyplot as plt
import torch

from encoders.spike.latency_encoder import LatencyEncoder
from data.dataset.random_dataset import RandomDataset, DataType, OutputType
from network.nodes.leaky_kernel import LeakyKernel
from network.nodes.den_kernel import DENKerenl
from tools.utils import SEQ_LEN
from common import ATTR, SPIKE_NS, MODEL_NS

DEFAULT_BATCH = 0

class RandomSpikePattern:
    """
    visualize the random spike pattern data
    """
    def __init__(self, batch_size: int = DEFAULT_BATCH) -> None:
        batch_size = batch_size if batch_size is not DEFAULT_BATCH else ATTR(MODEL_NS.BATCH_SIZE)
        
        T = ATTR(SPIKE_NS.T)
        dt = ATTR(SPIKE_NS.dt)
        self._dataset = RandomDataset(
            ATTR(MODEL_NS.NUM_INPUTS), 
            SEQ_LEN(T, dt),
            DataType.TRAIN,
            OutputType.TORCH, 
            LatencyEncoder(1))
    
    
    def single_spike_raster(self):
        """
        plots rasters of the random spike patterns
        """
        
        FIRST_IDX = 0
        data = self._dataset.get_raw(FIRST_IDX)
        
        data.plot()
        
    def lif_response(self):
        """
        plot the response of the spike
        """
        IDX = 1
        data = self._dataset[IDX]
        raw_data = self._dataset.get_raw(IDX)
        
        seq_len, n = data.shape
        
        kernel = LeakyKernel(n, scale=True)
        
        with torch.no_grad():
            response = kernel.forward(data)

        simulated_response = LeakyKernel.assimulate_response(raw_data, ATTR(SPIKE_NS.tau), ATTR(SPIKE_NS.dt))
        simulated_response = simulated_response.numpy()
        
        # Plot the results
        plt.figure()

        plt.plot(response, label='Output Spikes', color='red')
        plt.plot(simulated_response, label='Simulated Output Spikes', color='blue', linestyle=":")
        plt.title('Model Output Spikes')
        plt.xlabel('Time Steps')
        plt.ylabel('Spike')
        plt.legend()

        plt.tight_layout()
        plt.show()
        

    def den_response(self):
            """
            plot the response of the spike
            """
            IDX = 2
            data = self._dataset[IDX]
            raw_data = self._dataset.get_raw(IDX)
            
            seq_len, n = data.shape
            
            kernel = DENKerenl(n, scale=True)
            
            with torch.no_grad():
                response = kernel.forward(data)

            simulated_response = DENKerenl.assimulate_response(raw_data, ATTR(SPIKE_NS.tau_m), ATTR(SPIKE_NS.tau_s), ATTR(SPIKE_NS.dt))
            simulated_response = simulated_response.numpy()
            
            # Plot the results
            plt.figure()

            plt.plot(response, label='Output Spikes', color='red')
            plt.plot(simulated_response, label='Simulated Output Spikes', color='blue', linestyle=":")
            plt.title('Model Output Spikes')
            plt.xlabel('Time Steps')
            plt.ylabel('Spike')
            plt.legend()

            plt.tight_layout()
            plt.show()
        