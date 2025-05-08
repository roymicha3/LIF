from numba import cuda, njit, prange


#TODO: write a wrapper for the GPU kernel

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