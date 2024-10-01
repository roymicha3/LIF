"""
This module defines the DataSample class which encapsulates 
a single data item and provides a method to access it.
"""
import os
import torch
import pickle
import numpy as np

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
        Returns the shape of the data, supporting lists, NumPy arrays, and tensors.
        """
        data = self._data

        if isinstance(data, np.ndarray):
            return data.shape
        elif isinstance(data, torch.Tensor):
            return data.shape
        elif isinstance(data, list):
            def get_shape(lst):
                if isinstance(lst, list):
                    if len(lst) == 0:
                        return (0,)
                    first_elem_shape = get_shape(lst[0])
                    return (len(lst), *first_elem_shape)
                return ()
            
            return get_shape(data)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

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
        if not os.path.splitext(filename)[1]:  # If no extension is present
            filename += '.pkl'
        
        with open(filename, 'wb') as file:
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
