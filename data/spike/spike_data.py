"""
Container for spike data.
"""
import matplotlib.pyplot as plt

class SpikeData:
    """
    Container for spike data.
    """
    def __init__(self):
        """
        Initialize an empty spike data container.
        """
        self.spike_times = []

    def add_spike(self, time):
        """
        Add a spike at the specified time.

        Args:
            time (float): Time of the spike in milliseconds.
        """
        self.spike_times.append(time)

    def get_spike_times(self):
        """
        Get the list of spike times.

        Returns:
            list: List of spike times.
        """
        return self.spike_times

    def plot_spike_train(self, light_onset_time, light_offset_time):
        """
        Plot the spike train with shading for stimulus duration.

        Args:
            light_onset_time (float): Time when the stimulus starts.
            light_offset_time (float): Time when the stimulus ends.
        """

        _, ax = plt.subplots()
        ax.vlines(self.spike_times, 0, 1)
        ax.set_xlim([0, len(self.spike_times)])
        ax.set_xlabel('Time (ms)')
        ax.set_title('Neuronal Spike Times')
        ax.axvspan(light_onset_time, light_offset_time, alpha=0.5, color='greenyellow')
        plt.show()