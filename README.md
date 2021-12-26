# Numba CUDA intro

- GITHUB: https://github.com/daniel-vera-g/numba-cuda-intro

> Short Numba CUDA tutorial for the parallel computing class at the HKA

## Requirements

1. At least `miniconda` installed
2. Optionally a CUDA GPU

## Setup

1. Create environment and install dependencies: `conda env create --name parallele -f environment.yml`

If this step makes problems, just omit the `-f environment.yml` and install the necessary dependencies: `conda install numba numpy jupyter jupyter-lab`(After activating the environment)

2. Activate: `conda activate parallele`
3. Start jupyter lab: `jupyter-lab`
4. Have fun!

Additional notes:

- To save newly installed dependencies run `conda env export > environment.yml`
- If you don't have a CUDA capable GPU, you can activate the _simulation_ mode by starting Jupyterlab with the `NUMBA_ENABLE_CUDASIM=1` ENV: `NUMBA_ENABLE_CUDASIM=1 jupyter-lab`

## References

This tutorial uses the following references:

1. https://numba.pydata.org/numba-doc/dev/index.html
2. https://people.duke.edu/~ccc14/sta-663/CUDAPython.html
3. https://nyu-cds.github.io/python-numba/05-cuda/
4. Used image: https://www.researchgate.net/figure/Figure-2-Execution-model-of-a-CUDA-program-on-NVidias-GPU-Hierarchy-grid-blocks-and_fig2_321666991

Additional:

- Thread indexing cheat sheet: https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf
