from numba import cuda
import numpy as np
import math

@cuda.jit
def matmul(inputArr1, inputArr2, outputArr):
    """
    Do a Matrix multiplication: inputArr1 * inputArr2 = outputArr
    """
    # Get a two dimensional grid for calculations
    row, column = cuda.grid(2)
    
    # Check that we're in the boundaries
    # & not accessing prohibited memory
    if row < outputArr.shape[0] and column < outputArr.shape[1]:
        tmp = 0.
        for k in range(inputArr1.shape[1]):
            tmp += inputArr1[row, k] * inputArr2[k, column]
            outputArr[row, column] = tmp


# Create 2D-Arrays filled with 7 and 8s
# NOTE: Row <-> Column or Column <-> Row should be equal size
inputArr1 = np.full((12, 42), 7, float)
inputArr2 = np.full((42, 16), 8, float)

# Copy 2D-Arrays to device aka "GPU"
inputArr1_global_mem  = cuda.to_device(inputArr1)
inputArr2_global_mem  = cuda.to_device(inputArr2)

# Allocate mem on device for result
# Shape = non-equal size values from above
outputArr_global_mem = cuda.device_array((12,16))

# TODO how get values about threadsperblock?
threads_in_block = (16, 16)
blocks_per_grid_x = int(math.ceil(inputArr1.shape[0] / threads_in_block[0]))
blocks_per_grid_y = int(math.ceil(inputArr1.shape[1] / threads_in_block[1]))
blocks_in_grid = (blocks_per_grid_x, blocks_per_grid_y)

matmul[blocks_in_grid, threads_in_block](inputArr1_global_mem, inputArr2_global_mem, outputArr_global_mem)

# Copy result back to host aka. CPU
outputArr = outputArr_global_mem.copy_to_host()

print(outputArr)
