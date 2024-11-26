# Contributing to `diffopt`

Thank you for your interest in contributing to this project. All questions and ideas for improvement are welcome and can be made through opening an issue or pull request.

Before contributing, familiarize yourself with our resources:

- [Source Code](https://github.com/AlanPearl/diffopt)
- [Documentation](https://diffopt.readthedocs.io)

## Issues

You can open an [issue](https://github.com/AlanPearl/diffopt/issues) if you:

- Have encountered a bug or issue when using the software
- Would like to see a new feature
- Are seeking support that could not be resolved by reading the documentation

## Pull Requests

If you would like to directly submit your own change to the software, thank you! Here's how:

- Fork [this repository](https://github.com/AlanPearl/diffopt).
- Please remember to include a concise, self-contained unit test in your pull request. Ensure that all tests pass (see [Manual Testing](#manual-testing)).
- Open a [pull request](https://github.com/AlanPearl/diffopt/pulls).

## Manual Testing

Make sure you have installed diffopt as described in the [docs](https://diffopt.readthedocs.io/en/latest/installation.html). To run all tests from the main directory:

```bash
pip install pytest
pytest .
mpirun -n 2 pytest .
```

Note that unit tests requiring `mpi4py` installation are not automatically tested by GitHub workflows. Therefore, running these tests manually with `mpi4py` installed is necessary to assure that all tests pass.
