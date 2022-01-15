from numba import cuda
import numpy as np

# Our jit cuda enabled function 
@cuda.jit
def test_kernel(an_array):
    """
    TODO Add the code to be executed by CUDA here
    """

# Declare kernel:

# 1. Data array to work with
# Docs: "Return a new array of given shape and type, filled with ones."
dataArr = np.ones(256)

# 2. Define Thread values

# Block size depending on size data array, shared memory, supported hardware, ...
threads_in_block = 32
blocks_in_grid = (dataArr.size + (threads_in_block - 1))

# 3. "Call" Kernel
test_kernel[blocks_in_grid, threads_in_block](dataArr)

print(dataArr)
