"""
This module defines the DataSample class which encapsulates 
a single data item and provides a method to access it.
"""

class DataSample:
    """
    This class represents a data sample.
    It encapsulates a single data item and provides a method to access it.
    """
    def __init__(self, data, label = None):
        self.__data = data
        self.__label = label

    def get(self):
        """
        Returns the data.
        """
        return self.__data

    def plot(self):
        """
        plots the data.
        """
        raise NotImplementedError
    
    def get_label(self):
        """
        Returns the label.
        """
        return self.__label

    def __str__(self):
        return f"DataSample({self.__data})"
