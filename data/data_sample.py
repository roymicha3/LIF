"""
This module defines the DataSample class which encapsulates 
a single data item and provides a method to access it.
"""

class DataSample:
    """
    This class represents a data sample.
    It encapsulates a single data item and provides a method to access it.
    """
    def __init__(self, data):
        self._data = data

    def get(self):
        """
        Returns the data.
        """
        return self._data

    def __str__(self):
        return f"DataSample({self._data})"
