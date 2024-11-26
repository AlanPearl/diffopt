# diffopt
Parallelization and optimization of differentiable and many-parameter models

## Author
- Alan Pearl

## Documentation
Online documentation is available at [diffopt.readthedocs.io](https://diffopt.readthedocs.io/en/latest).

## Manual Testing
Unit tests requiring `mpi4py` installation are not automatically tested by GitHub workflows. To run all tests, install `mpi4py` locally, and run `pytest` from the root directory. Additionally, all tests must pass with `mpirun -n 4 pytest` etc. (up to the maximum number of tasks that can be run on your machine).