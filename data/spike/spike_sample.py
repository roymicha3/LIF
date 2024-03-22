"""
This module defines the SpikeSample class which encapsulates
"""
from typing import List

from data.data_sample import DataSample
from data.spike.spike_data import SpikeData

class SpikeSample(DataSample):
    """
    This class represents a spike sample.
    It encapsulates a single spike item and provides a method to access it.
    """
    def __init__(self, data: List[SpikeData], attr: dict) -> None:
        super().__init__(data)
        self._num_of_neurons = len(data)
        self._attr = attr

    def __str__(self):
        return f"SpikeSample({self._data})"
