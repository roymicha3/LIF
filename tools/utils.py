"""
Utility functions.
"""
import numpy as np

def max_with_default(arr, default_value = 0):
    """
    Return the maximum value of the array or the default value if the array is empty.
    """
    return np.max(arr) if arr.size > 0 else default_value
