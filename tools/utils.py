"""
Utility functions.
"""
import numpy as np

def max_with_default(arr, default_value = 0):
    """
    Return the maximum value of the array or the default value if the array is empty.
    """
    return np.max(arr) if arr.size > 0 else default_value


def poisson_events(k: int, T: int) -> np.array:
    """
    samples k events from a poisson process with rate T
    """
    # Generate a random integer between 0 and T-1 (inclusive)
    event_times = np.random.randint(0, T, size=k)
    return event_times

def calculate_sequence_len(T: int, dt: int):
    """
    calculates the number of time samples within the measured window
    """
    return int(T / dt)

SEQ_LEN = calculate_sequence_len

