from numba import cuda

cuda.select_device(0) # Default = fastest

# cuda.select_device(1) => Having more than one GPU available

print(f"Avvailable GPUs: {cuda.gpus}")
print(f"GPU currently in use: {cuda.gpus.current}")
