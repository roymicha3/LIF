import math
import torch
from numba import cuda


################################# LIF Kernel #################################

@cuda.jit
def numba_lif(x, v_prev, v_next, spike, v_threshold, beta):
    i, j = cuda.grid(2)
    if i < x.shape[0] and j < x.shape[1]:
        v = v_prev[i,j] * beta + x[i,j]
        if v >= v_threshold:
            spike[i,j] = 1.0
        
        v_next[i,j] = v


@cuda.jit
def numba_lif_hard_reset(x, v_prev, v_next, spike, v_threshold, beta):
    i, j = cuda.grid(2)
    if i < x.shape[0] and j < x.shape[1]:
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
    assert x.device.type == "cuda"
    assert v_prev.device.type == "cuda"
    
    v_next = torch.zeros_like(x)
    spike = torch.zeros_like(x)
    
    threadsperblock = (16, 16)  # 256 total threads
    blockspergrid_x = math.ceil(x.size()[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(x.size()[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    if hard_reset:
        numba_lif_hard_reset[blockspergrid, threadsperblock](x, v_prev, v_next, spike, v_threshold, beta)
    else:
        numba_lif[blockspergrid, threadsperblock](x, v_prev, v_next, spike, v_threshold, beta)

    return v_next, spike


def den(
    x: torch.Tensor,
    i_prev: torch.Tensor,
    v_prev: torch.Tensor,
    v_threshold: float,
    alpha: float,
    beta: float,
    hard_reset: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Wraps the Numba-based LIF function for PyTorch tensors.
    """
    assert x.device.type == "cuda"
    assert v_prev.device.type == "cuda"
    
    v_next = torch.zeros_like(x)
    i_next = torch.zeros_like(x)
    
    i_spike = torch.zeros_like(x)
    v_spike = torch.zeros_like(x)
    
    threadsperblock = (16, 16)  # 256 total threads
    blockspergrid_x = math.ceil(x.size()[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(x.size()[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    numba_lif[blockspergrid, threadsperblock](x, i_prev, i_next, i_spike, v_threshold, beta)
    
    if hard_reset:
        numba_lif_hard_reset[blockspergrid, threadsperblock](i_next, v_prev, v_next, v_spike, v_threshold, alpha)
    else:
        numba_lif[blockspergrid, threadsperblock](i_next, v_prev, v_next, v_spike, v_threshold, alpha)

    return i_next, i_spike, v_next, v_spike