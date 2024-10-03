
"""
Collection of utility functions, mostly relating to simple gradient
descent, which is not really recommended to be used over adam.
"""
from typing import NamedTuple, Union

import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import qmc

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
    Comm = MPI.Comm
    Intracomm = MPI.Intracomm
except ImportError:
    MPI = COMM = None
    Comm = Intracomm = type(None)
    RANK = 0
    N_RANKS = 1

try:
    if RANK:
        raise ImportError("Only show progress bar on RANK=0 task")
    from tqdm import auto as tqdm
except ImportError:
    tqdm = None


__all__ = ["simple_grad_descent", "GradDescentResult",
           "latin_hypercube_sampler", "scatter_nd"]


def trange_no_tqdm(n, desc=None):
    return range(n)


def trange_with_tqdm(n, desc=None):
    return tqdm.trange(n, desc=desc)


trange = trange_no_tqdm if tqdm is None else trange_with_tqdm


class GradDescentResult(NamedTuple):
    loss: jnp.ndarray
    params: jnp.ndarray
    aux: Union[jnp.ndarray, list]


def latin_hypercube_sampler(xmin, xmax, n_dim, num_evaluations,
                            seed=None, optimization=None):
    xmin = np.zeros(n_dim) + xmin
    xmax = np.zeros(n_dim) + xmax
    sampler = qmc.LatinHypercube(n_dim, seed=seed, optimization=optimization)
    unit_hypercube = sampler.random(num_evaluations)
    return qmc.scale(unit_hypercube, xmin, xmax)


def scatter_nd(array, axis=0, comm=COMM, root=0):
    """Scatter n-dimensional array from root to all ranks"""
    ans: np.ndarray = np.array([])
    if comm.rank == root:
        splits = np.array_split(array, comm.size, axis=axis)
        for i in range(comm.size):
            if i == root:
                ans = splits[i]
            else:
                comm.send(splits[i], dest=i)
    else:
        ans = comm.recv(source=root)
    return ans


def simple_grad_descent(
    loss_func,
    guess,
    nsteps,
    learning_rate,
    loss_and_grad_func=None,
    grad_loss_func=None,
    has_aux=False,
    **kwargs,
):
    if loss_and_grad_func is None:
        if grad_loss_func is None:
            loss_and_grad_func = jax.value_and_grad(
                loss_func, has_aux=has_aux, **kwargs)
        else:
            def explicit_loss_and_grad_func(params):
                return (loss_func(params), grad_loss_func(params))
            loss_and_grad_func = explicit_loss_and_grad_func

    # Create our mpi4jax token with a dummy broadcast
    def loopfunc(state, _x):
        grad, params = state
        params = jnp.asarray(params)

        # Evaluate the loss and gradient at given parameters
        (loss, grad), aux = loss_and_grad_func(params), None
        if has_aux:
            (loss, aux), grad = loss, grad
        y = (loss, params, aux)

        # Calculate the next parameters to evaluate (no need to broadcast this)
        params = params - learning_rate * grad
        # params = broadcast(params, root=0)
        state = grad, params
        return state, y

    # The below is equivalent to lax.scan without jitting
    # ===================================================
    initstate = (0.0, guess)
    loss, params, aux = [], [], []
    for x in trange(nsteps, desc="Simple Gradient Descent Progress"):
        initstate, y = loopfunc(initstate, x)
        loss.append(y[0])
        params.append(y[1])
        aux.append(y[2])
    loss = jnp.array(loss)
    params = jnp.array(params)
    if has_aux:
        try:
            aux = jnp.array(aux)
        except TypeError:
            pass
    ##################################

    return GradDescentResult(loss=loss, params=params, aux=aux)
