"""
This class represents spike data for a set of neurons.
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from common import ATTR

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
        self.__T = ATTR().get_attr('T')


    def __copy__(self):
        """
        Implement the copy operation.
        :return: A copy of the current SpikeData object.
        """
        return SpikeData(self.__neuron_index, self.__spike_times)


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
