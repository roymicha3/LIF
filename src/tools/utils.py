"""
Utility functions.
"""
import os
import numpy as np
from typing import List

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

def get_prefix_files(directory: str, word: str, max_depth: int=5) -> List[str]:
        result = []

        def search_dir(current_dir, current_depth):
            if current_depth > max_depth:
                return
            try:
                for entry in os.scandir(current_dir):
                    if entry.is_file() and word in entry.name:
                        result.append(entry.path)
                    
                    elif entry.is_dir():
                        search_dir(entry.path, current_depth + 1)
            
            except PermissionError:
                pass

        search_dir(directory, 0)
        return result


SEQ_LEN = calculate_sequence_len

