"""
this file contains the utility functions for the spike encoder
"""
import numpy as np

def poisson_events(k: int, T: int) -> np.array:
    """
    samples k events from a poisson process with rate T
    """
    # Generate a random integer between 0 and T-1 (inclusive)
    event_times = np.random.randint(0, T, size=k)
    return event_times
