from numba import cuda
import numpy as np
import os

os.environ["NUMBA_ENABLE_CUDASIM"] = "1" # Activate the simulator

@cuda.jit
def vec_add(vecIn1, vecIn2, vecOut):
    """
    Simple one dimensional Vector addition.
    Debugging on a specific index.
    """
    
    threadX = cuda.threadIdx.x
    blockX = cuda.blockIdx.x
    blockDimX = cuda.blockDim.x

    # Stop the debugger at thread 1 in block 3
    if threadX == 1 and blockX == 3:
        from pdb import set_trace

        set_trace()
    
    i = blockX * blockDimX + threadX
    vecOut[i] = vecIn1[i] + vecIn2[i]
    
    
vec1 = np.ones((16))
vec2 = np.ones((16))
vec3 = np.ones((16))

vec_add[4,4](vec1, vec2, vec3)
