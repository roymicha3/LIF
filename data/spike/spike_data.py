"""
This class represents spike data for a set of neurons.
"""
import matplotlib
import seaborn as sns
import matplotlib.axes
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from typing_extensions import override

# TODO: add spike values to the data as optional!

class SpikeData:
    """
    Initialize a SpikeData object.

    Parameters:
    neuron_indices (list of int): List of neuron indices.
    spike_times (list of int): List of spike times of the neuron.
    """

    def __init__(self, env_config: DictConfig, neuron_index, spike_times: list[int]):

        self.neuron_index = neuron_index
        self.spike_times = spike_times
        self.env_config = env_config
        self.T = env_config.T


    def __copy__(self):
        """
        Implement the copy operation.
        :return: A copy of the current SpikeData object.
        """
        return SpikeData(self.env_config, self.neuron_index, self.spike_times)
    
    def get_index(self):
        return self.neuron_index

    def get_spike_times(self):
        return self.spike_times

    def mean_firing_rate(self) -> float:
        """
        Return the mean firing rate of the neuron.
        """
        return len(self.spike_times) / self.T
    
    def silence(self, cutoff: int):
        """
        Silence all spikes above the given cutoff
        """
        self.spike_times = [t for t in self.spike_times if t <= cutoff]


    def plot_spike_train(self) -> None:
        """
        Plot the spike train for a single neuron.
        """
        plt.figure(figsize=(10, 6))
        sns.rugplot(self.spike_times, height=0.5)
        plt.xlim([0, self.T])  # set x-axis limits
        plt.title(f'Spike Train for Neuron {self.neuron_index}', fontsize=14)
        plt.xlabel('Time (s)', fontsize=12)
        plt.yticks([])
        plt.show()
    
    @override   
    def plot(self, plot: matplotlib.axes.Axes) -> matplotlib.figure.Figure:
        """
        this function adds
        """
        return add_neuron_raster(self, plot)

def add_neuron_raster(data: SpikeData, plot: matplotlib.figure.Figure) -> matplotlib.figure.Figure:
    """
    Adds the raster plot of a specific neuron to an existing raster plot.
    """
    # Extract spike times for the specified neuron
    neuron_spikes = data.get_spike_times()
    neuron_index = data.get_index()

    # Plot the spikes for the neuron
    plot.eventplot(neuron_spikes, lineoffsets=neuron_index, colors='k', linewidths=0.5)

    return plot
