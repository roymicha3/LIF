"""
this file contains the utility functions for the spike encoder
"""
import numpy as np
from common import *

def poisson_events(k: int, T: int) -> np.array:
    """
    samples k events from a poisson process with rate T
    """
    # Generate a random integer between 0 and T-1 (inclusive)
    event_times = np.random.randint(0, T, size=k)
    return event_times

def calculate_time_samples(T: int, dt: int):
    """
    calculates the number of time samples within the measured window
    """
    return int(T / dt)

NUM_TIME_SAMPLES = calculate_time_samples
