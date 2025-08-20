import torch
import numpy as np
import multiprocessing
from numba import njit, prange, set_num_threads


# Use only quarter of available cores
max_threads = max(1, multiprocessing.cpu_count() // 4)
set_num_threads(max_threads)


@njit(parallel=True)
def numba_den(x, i_prev, v_prev, i_next, i_spike, v_next, v_spike, 
              v_threshold, alpha, beta, hard_reset):
    """
    Fused dendritic neuron kernel - processes both current and voltage dynamics in one pass.
    """
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            # Stage 1: Current dynamics
            i_val = i_prev[i,j] * beta + x[i,j]
            if i_val >= v_threshold:
                i_spike[i,j] = 1.0
            i_next[i,j] = i_val
            
            # Stage 2: Voltage dynamics (using updated current)
            v_val = v_prev[i,j] * alpha + i_val
            if v_val >= v_threshold:
                v_spike[i,j] = 1.0
                if hard_reset:
                    v_val = 0.0
            v_next[i,j] = v_val


@njit(parallel=True)
def numba_lif(x, v_prev, v_next, spike, v_threshold, beta):
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            v = v_prev[i,j] * beta + x[i,j]
            if v >= v_threshold:
                spike[i,j] = 1.0
            v_next[i,j] = v


@njit(parallel=True)
def numba_lif_hard_reset(x, v_prev, v_next, spike, v_threshold, beta):
    for i in prange(x.shape[0]):
        for j in prange(x.shape[1]):
            v = v_prev[i,j] * beta + x[i,j]
            if v >= v_threshold:
                spike[i,j] = 1.0
                v = 0.0
            v_next[i,j] = v


def lif(
    x: torch.Tensor,
    v_prev: torch.Tensor,
    v_threshold: float,
    beta: float,
    hard_reset: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Wraps the Numba-based LIF function for PyTorch tensors.
    """
    assert x.device == torch.device("cpu")
    assert v_prev.device == torch.device("cpu")

    x_np = x.detach().numpy()
    v_prev_np = v_prev.detach().numpy()
    v_next_np = np.zeros_like(x_np)
    spike_np = np.zeros_like(x_np)

    if hard_reset:
        numba_lif_hard_reset(x_np, v_prev_np, v_next_np, spike_np, v_threshold, beta)
    else:
        numba_lif(x_np, v_prev_np, v_next_np, spike_np, v_threshold, beta)

    v_next = torch.from_numpy(v_next_np)
    spike = torch.from_numpy(spike_np)

    return v_next, spike


def den(
    x: torch.Tensor,
    i_prev: torch.Tensor,
    v_prev: torch.Tensor,
    v_threshold: float,
    alpha: float,
    beta: float,
    hard_reset: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Efficient dendritic neuron function with fused kernel.
    Processes both current and voltage dynamics in a single pass.
    """
    assert x.device == torch.device("cpu")
    assert i_prev.device == torch.device("cpu")
    assert v_prev.device == torch.device("cpu")

    # Convert to numpy arrays
    x_np = x.detach().numpy()
    i_prev_np = i_prev.detach().numpy()
    v_prev_np = v_prev.detach().numpy()
    
    # Pre-allocate all output arrays
    i_next_np = np.empty_like(x_np)
    i_spike_np = np.zeros_like(x_np)
    v_next_np = np.empty_like(x_np)
    v_spike_np = np.zeros_like(x_np)
    
    # Single fused kernel call - processes both stages
    numba_den(x_np, i_prev_np, v_prev_np, i_next_np, i_spike_np, 
              v_next_np, v_spike_np, v_threshold, alpha, beta, hard_reset)

    # Convert back to PyTorch tensors
    return (torch.from_numpy(i_next_np), 
            torch.from_numpy(i_spike_np), 
            torch.from_numpy(v_next_np), 
            torch.from_numpy(v_spike_np))