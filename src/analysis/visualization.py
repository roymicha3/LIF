"""
Here we visualize our data and results
"""
import os
import torch
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from data.dataset import Dataset, DataType, OutputType, DatasetFactory
from encoders.encoder_factory import EncoderFactory

def load_dataset(
    dataset_config: DictConfig,
    env_config: DictConfig,
    type_: DataType = DataType.TRAIN) -> Dataset:
    
    encoder_config = dataset_config.encoder
    encoder = EncoderFactory.create(encoder_config.type, encoder_config, env_config)
    dataset = DatasetFactory.create(
        dataset_config.type, dataset_config, 
        type_, 
        OutputType.TORCH, 
        encoder)
    
    return dataset

class RandomSpikePattern:
    """
    visualize the random spike pattern data
    """
    def __init__(self, config: DictConfig, env_config: DictConfig) -> None:
        self.config = config
        self.env_config = env_config
        
        self._dataset = load_dataset(config.data, env_config, DataType.TRAIN)
    
    
    def single_spike_raster(self):
        """
        plots rasters of the random spike patterns
        """
        
        FIRST_IDX = 0
        data = self._dataset.get_raw(FIRST_IDX)
        
        data.plot()
        
    def lif_response(self, tau: float):
        """
        plot the response of the spike
        """
        from network.kernel.leaky_kernel import LeakyKernel
        
        IDX = 1
        data, label = self._dataset[IDX]
        raw_data = self._dataset.get_raw(IDX)
        
        seq_len, n = data.shape
        
        kernel = LeakyKernel(self.env_config, n, tau)
        
        with torch.no_grad():
            response = kernel.forward(data)

        simulated_response = LeakyKernel.assimulate_response(raw_data, tau, self.env_config.dt)
        simulated_response = simulated_response.numpy()
        
        # Plot the results
        plt.figure()

        plt.plot(response, label='Output Spikes', color='red')
        plt.plot(simulated_response, label='Simulated Output Spikes', color='blue', linestyle=":")
        plt.title('Model Output Spikes')
        plt.xlabel('Time Steps')
        plt.ylabel('Voltage')
        plt.legend()

        plt.tight_layout()
        plt.show()
        
    def den_response(self, tau_m: float, tau_s: float):
            """
            plot the response of the spike
            """
            from network.kernel.den_kernel import DENKernel
            IDX = 2
            data, label = self._dataset[IDX]
            raw_data = self._dataset.get_raw(IDX)
            
            seq_len, n = data.shape
            
            kernel = DENKernel(self.env_config, n, tau_m, tau_s)
            
            with torch.no_grad():
                response = kernel.forward(data)

            simulated_response = DENKernel.assimulate_response(raw_data, tau_m, tau_s, self.env_config.dt)
            simulated_response = simulated_response.numpy()
            
            # Plot the results
            plt.figure()

            plt.plot(response, label='Output Spikes', color='red')
            plt.plot(simulated_response, label='Simulated Output Spikes', color='blue', linestyle=":")
            plt.title('Model Output Spikes')
            plt.xlabel('Time Steps')
            plt.ylabel('Voltage')
            plt.legend()

            plt.tight_layout()
            plt.show()
        
    def simple_network_response(self):
        """
        plot the response for a random spike input of a given network
        """
        # TODO: implement this
        pass
        