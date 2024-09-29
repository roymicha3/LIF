"""
This module defines the DataSample class which encapsulates 
a single data item and provides a method to access it.
"""
import os
import pickle

class DataSample:
    """
    This class represents a data sample.
    It encapsulates a single data item and provides a method to access it.
    """
    def __init__(self, data, label = None):
        self._data = data
        self._label = label

    def get(self):
        """
        Returns the data.
        """
        return self._data
    
    def to_torch(self):
        """
        Returns the data as a torch tensor
        """
        pass
    
    def to_numpy(self):
        """
        Returns the data as a numpy array
        """
        pass
    
    def shape(self):
        """
        Returns the shape of the data
        """
        return self._data.shape

    def plot(self):
        """
        plots the data.
        """
        raise NotImplementedError
    
    def get_label(self):
        """
        Returns the label.
        """
        return self._label
    
    def __getitem__(self, index):
        """
        Allows access to the data by index.
        """
        return self._data[index]

    def __str__(self):
        return f"DataSample({self._data})"
    
    def serialize(self, filename):
        """
        Serializes the DataSample object to a file.
        """
        with open(f"{filename}.pkl", 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def deserialize(cls, filename):
        """
        Deserializes a DataSample object from a file.
        """
        if not os.path.splitext(filename)[1]:  # If no extension is present
            filename += '.pkl'
        
        with open(filename, 'rb') as file:
            return pickle.load(file)
