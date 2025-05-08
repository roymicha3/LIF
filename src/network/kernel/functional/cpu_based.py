import torch
import numpy as np
from numba import njit, prange


@njit(parallel=True)
def numba_lif(x, v_prev, v_next, spike, v_threshold, beta):
    for i in prange(v_prev.shape[0]):
        for j in prange(v_prev.shape[1]):
            v = v_prev[i,j] * beta + x[i,j]
            if v >= v_threshold:
                spike[i,j] = 1.0
            
            v_next[i,j] = v


@njit(parallel=True)
def numba_lif_hard_reset(x, v_prev, v_next, spike, v_threshold, beta):
    for i in prange(v_prev.shape[0]):
        for j in prange(v_prev.shape[1]):
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