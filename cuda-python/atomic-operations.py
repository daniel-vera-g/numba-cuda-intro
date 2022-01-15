from numba import cuda
import numpy as np

@cuda.jit
def find_max_value(result, input):
    """
        Find the maximum value of the input array
    """
    
    i = cuda.grid(1)
        
    # Is i the new minimum value?
    cuda.atomic.max(result, 0, input[i])
    
# Array of random values
inArray = np.random.rand(16384)
result = np.zeros(1, dtype=np.float64)

find_max_value[256, 64](result, inArray)

print(f"Maximum value in Array using atomic operations:\n{result[0]}")
print(f"Maximum value in Array using simple python:\n{max(inArray)}")
