# Numba CUDA intro

> Numba CUDA tutorial for the parallel computing class at the HKA
>
> **Up to date version: https://github.com/daniel-vera-g/numba-cuda-intro**

1. _TLDR; 👉_: [Online Quickstart](https://github.com/daniel-vera-g/numba-cuda-intro/blob/master/cuda-notebook/numba_cuda_tutorial.ipynb)
2. Explore in online editor(Google account needed to run 💡): [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/daniel-vera-g/numba-cuda-intro)

---

1. [Requirements](#requirements)
2. [Setup](#setup)
   - [Additional notes](#additional-notes)
3. [Contents](#contents)
4. [References](#references)

## Requirements

1. Python 3 with at least [`miniconda`](https://docs.conda.io/en/latest/miniconda.html) installed
2. Optionally a CUDA GPU

## Setup

**Automated:**

1. Create conda environment and install packages if needed: `./run.sh`
2. Same as above and start jupyter notebook: `./run.sh --jupyter`

**Manual:**

1. Create an environment and install the dependencies: `conda env create --name parallele -f environment.yml`

> **NOTE**: If this step makes problems, just omit the `-f environment.yml` and install the necessary dependencies manually after activating the environment: `conda install numba numpy jupyterlab`

2. If not already done, activate the environment: `conda activate parallele`
3. Start jupyter notebook: `jupyter-notebook`

### Additional notes

- To save newly installed dependencies, run `conda env export > environment.yml`
- If you don't have a CUDA capable GPU, you can activate the _simulation_ mode by starting Jupyter notebook with the `NUMBA_ENABLE_CUDASIM=1` ENV: `NUMBA_ENABLE_CUDASIM=1 jupyter-notebook`

## Contents

- [Basics](./cuda-notebook/numba_cuda_tutorial.ipynb)

**CUDA concepts in Numba:**

- What Kernels are and how they work: [Kernels](./cuda-notebook/kernels.ipynb)
- How to manage memory when doing operations: [Memory management](./cuda-notebook/memory-management.ipynb)
- How to debug Numba code: [Debugging](./cuda-notebook/cuda-debugger.ipynb)
- Other useful CUDA features in Numba: [Other Numba features](./cuda-notebook/other-cuda-features.ipynb)

## References

This tutorial uses the following references:

1. https://numba.pydata.org/numba-doc/dev/index.html
2. https://people.duke.edu/~ccc14/sta-663/CUDAPython.html
3. https://nyu-cds.github.io/python-numba/05-cuda/
4. Used image: https://www.researchgate.net/figure/Figure-2-Execution-model-of-a-CUDA-program-on-NVidias-GPU-Hierarchy-grid-blocks-and_fig2_321666991

Additional:

- Thread indexing cheat sheet: https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf
