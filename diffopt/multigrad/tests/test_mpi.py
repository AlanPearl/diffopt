"""
This test can be run explicitly in an MPI environment.
It should work for any number of processes, e.g.
`pytest test_mpi.py`, `mpiexec -n 2 pytest test_mpi.py`,
and `mpiexec -n 10 pytest test_mpi.py` all must pass
(the --with-mpi flag shouldn't have any effect)
"""
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
import jax.numpy as jnp

import unittest
from ... import multigrad
from .smf_example import smf_grad_descent as sgd


@unittest.skipIf(MPI is None, "MPI must be installed to run this test")
def test_reduce_sum():
    rank, size = MPI.COMM_WORLD.rank, MPI.COMM_WORLD.size
    # Set value equal to the rank of the process
    value = jnp.array(rank)

    # Reduce the sum of the values across all ranks
    result = multigrad.reduce_sum(value)

    # Gather the results from all processes
    gathered_results = jnp.array(MPI.COMM_WORLD.allgather(result))

    if not rank:
        # Perform testing only on the rank 0 process
        # Ensure that all results are the same and give the expected value
        expected_result = jnp.arange(size).sum()
        assert all(x == expected_result for x in gathered_results), \
            f"Rank{rank} gathered {gathered_results}, " \
            f"but expected = {expected_result}"


def test_grad_descent_methods():
    num_halos = 10_000
    data = dict(
        log_halo_masses=jnp.log10(sgd.load_halo_masses(num_halos)),
        smf_bin_edges=jnp.linspace(9, 10, 11),
        volume=10.0 * num_halos,  # Mpc^3/h^3
        target_sumstats=jnp.array([  # SMF at truth: params=(-2.0, 0.2)
            2.30178721e-02, 1.69728529e-02, 1.16054425e-02, 7.10532581e-03,
            3.77187086e-03, 1.69136131e-03, 6.28149020e-04, 1.90466686e-04,
            4.66692982e-05, 9.17260695e-06]),
    )
    model = sgd.MySMFModel(aux_data=data)

    truth = sgd.ParamTuple(log_shmrat=-2.0, sigma_logsm=0.2)
    true_sumstats = jnp.array(model.calc_sumstats_from_params(truth))
    true_gradloss = jnp.array(model.calc_dloss_dparams(truth))
    assert jnp.allclose(
        data["target_sumstats"], true_sumstats)

    # Calculate grad(loss) with the more memory efficient method
    loss, dloss_dparams = model.calc_loss_and_grad_from_params(truth)

    # Make sure it produces the same result as the memory intensive versions
    assert jnp.allclose(loss, model.calc_loss_from_params(truth))
    assert jnp.allclose(dloss_dparams, true_gradloss)
    assert jnp.allclose(true_gradloss, 0.0, atol=1e-2)

    # Run each gradient descent optimizer and check that the final loss
    # is not far from zero and params are close to truth
    # ==================================================================

    # Simple Grad Descent optimizer
    gd_iterations = model.run_simple_grad_descent(guess=truth, nsteps=2)
    gd_loss, gd_params = gd_iterations.loss, gd_iterations.params
    assert gd_loss[-1] < 1e-2, f"SimpleGD loss too high: {gd_loss[-1]}"
    assert jnp.allclose(gd_params[-1], jnp.array([*truth]), atol=1e-2), \
        f"SimpleGD params too far from truth: {gd_params[-1]} vs {truth}"

    # Adam optimizer
    adam_params, adam_loss = model.run_adam(guess=truth, nsteps=2)
    assert adam_loss[-1] < 1e-2, f"Adam loss too high: {adam_loss[-1]}"
    assert jnp.allclose(adam_params[-1], jnp.array([*truth]), atol=1e-2), \
        f"Adam params too far from truth: {adam_params[-1]} vs {truth}"

    # BFGS optimizer
    # if MPI.COMM_WORLD.rank == 0:
    #     import pdb
    #     pdb.set_trace()
    # MPI.COMM_WORLD.barrier()
    bfgs_params, bfgs_loss, res = model.run_bfgs(guess=truth, maxsteps=2)
    assert res.success, "BFGS did not converge"
    assert bfgs_loss[-1] < 1e-2, f"BFGS loss too high: {bfgs_loss[-1]}"
    assert jnp.allclose(bfgs_params[-1], jnp.array([*truth]), atol=1e-2), \
        f"BFGS params too far from truth: {bfgs_params[-1]} vs {truth}"
