#!/bin/sh

# Be able to use conda in shell script
shell_name=`basename $SHELL`
eval "$(conda shell.$shell_name hook)"

if ! conda env list | grep -q "^parallele"; then
  conda create --name parallele && \
    echo "Created the 'parallele' environment" && \
    conda install numba numpy jupyterlab && \
    echo "Installed numba, numpy, jupyterlab..."
    else
      echo "The 'parallele' environment already exists...Proceeding with actiavtion" && \
        conda activate parallele && \
        echo "Activated the 'parallele' environment"
fi

wait

# Accept option to activate jupyter notebook
if [ "$1" = "--jupyter" ]; then
  echo "Activating jupyter notebook" && \
    # If no GPU availabl, set NUMBA_ENABLE_CUDASIM=1
    jupyter-notebook ./cuda-notebook
fi

