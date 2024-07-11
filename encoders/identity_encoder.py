"""
This module contains the IdentityEncoder class.
"""
import numpy as np

from encoders.encoder import Encoder

class IdentityEncoder(Encoder):
    """
    This class represents an encoder that does not encode the data.
    """
    def __init__(self) -> None:
        pass

    def __call__(self, data: np.array):
        """
        Returns the data as is.
        """
        return data
