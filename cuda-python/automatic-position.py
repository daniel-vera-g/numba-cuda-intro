import numpy as np

@cuda.jit
def automated_kernel(an_array):
    # Docs: "Return the absolute position of the current thread in the entire grid of blocks."
    # `ndim` => Number dimensions
    position = cuda.grid(1)
        
    # Same as above
    if position < an_array.size:
        an_array[position] *= 2 

# Same as above:
dataArr = np.ones(256)
threads_in_block = 32
blocks_in_grid = (dataArr.size + (threads_in_block - 1))

automated_kernel[blocks_in_grid, threads_in_block](dataArr)

print(dataArr)
