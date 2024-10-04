Installation Instructions
=========================

Installation
------------
``pip install diffopt``

Prerequisites
-------------
- Tested on Python versions: ``python=3.9-12``
- MPI (several implementations available - see https://pypi.org/project/mpi4py)
- JAX (GPU install available - see https://jax.readthedocs.io/en/latest/installation.html)

Example installation with conda env:
++++++++++++++++++++++++++++++++++++

.. code-block:: bash

    conda create -n py312 python=3.12
    conda activate py312
    conda install -c conda-forge mpi4py jax
    pip install diffopt
