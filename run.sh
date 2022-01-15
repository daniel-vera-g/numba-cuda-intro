#!/bin/sh

shell_name=`basename $SHELL`


eval "$(conda shell.$shell_name hook)"

if ! conda env list | grep -q "^parallele"; then
  conda create --name parallele && \
    echo "Created the 'parallele' environment" && \
    conda install numba numpy jupyter jupyter-notebook && \
    echo "Installed numba, numpy, jupyter, and jupyter-notebook...Proceeding with notebook setup" && \
    NUMBA_ENABLE_CUDASIM=1 jupyter-notebook
    else
      echo "The 'parallele' environment already exists...Proceeding with actiavtion" && \
        conda activate parallele && \
        echo "Activated the 'parallele' environment"  && \
              NUMBA_ENABLE_CUDASIM=1 jupyter-notebook
fi

