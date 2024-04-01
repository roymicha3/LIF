"""
This class represents spike data for a set of neurons.
"""
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from typing_extensions import override

from common import ATTR, SPIKE_NS

class SpikeData:
    """
    Initialize a SpikeData object.

    Parameters:
    neuron_indices (list of int): List of neuron indices.
    spike_times (list of int): List of spike times of the neuron.
    """

    def __init__(self, neuron_index, spike_times: list[int]):

        self.__neuron_index = neuron_index
        self.__spike_times = spike_times
        self.__T = ATTR(SPIKE_NS.T)


    def __copy__(self):
        """
        Implement the copy operation.
        :return: A copy of the current SpikeData object.
        """
        return SpikeData(self.__neuron_index, self.__spike_times)
    
    def get_index(self):
        return self.__neuron_index

    def get_spike_times(self):
        return self.__spike_times

    def mean_firing_rate(self) -> float:
        """
        Return the mean firing rate of the neuron.
        """
        return len(self.__spike_times) / self.__T


    def plot_spike_train(self) -> None:
        """
        Plot the spike train for a single neuron.
        """
        plt.figure(figsize=(10, 6))
        sns.rugplot(self.__spike_times, height=0.5)
        plt.xlim([0, self.__T])  # set x-axis limits
        plt.title(f'Spike Train for Neuron {self.__neuron_index}', fontsize=14)
        plt.xlabel('Time (s)', fontsize=12)
        plt.yticks([])
        plt.show()
    
    @override   
    def plot(self, plot: matplotlib.figure.Figure) -> matplotlib.figure.Figure:
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

    # Create a subplot for the new neuron's raster plot
    ax = plot.add_subplot(111)

    # Plot the spikes for the neuron
    ax.eventplot(neuron_spikes, lineoffsets=neuron_index, colors='k', linewidths=0.5)

    # Customize the plot (you can adjust these settings as needed)
    ax.set_title(f"Raster Plot for Neuron {neuron_index}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron Index")
    ax.set_yticks([])  # Hide y-axis ticks

    return plot
