import numpy as np

def position_kernel(an_array):
    # Let's get the thread and corresponding block
    tt = cuda.threadIdx.x # Aka. X-Dimension
    tb = cuda.blockIdx.x # Aka. Y-Dimension
    
    # "Size" aka. width of Block: Number threads in Block
    bs = cuda.blockDim.x
    
    position = tt + tb * bs

    if position < an_array.size:
        an_array[position] *= 2 # Just double size as example

# Same as above:
dataArr = np.ones(256)
threads_in_block = 32
blocks_in_grid = (dataArr.size + (threads_in_block - 1))

position_kernel[blocks_in_grid, threads_in_block](dataArr)

print(dataArr)@cuda.jit
