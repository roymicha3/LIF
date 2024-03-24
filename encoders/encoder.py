"""
Abstract class for encoders.
"""
import numpy as np

class Encoder:
    """
    This class represents an encoder.
    """
    def __init__(self) -> None:
        pass

    def __call__(self, data: np.array):
        """
        encode data into a format that can be used by the model.
        """
        raise NotImplementedError

    def one_hot(self, labels, max_val=10):
        """
        encode labels into one hot vectors
        """
        encoded = np.zeros((len(labels), max_val), dtype=int)
        encoded[np.arange(len(labels)), labels] = 1
        return encoded
