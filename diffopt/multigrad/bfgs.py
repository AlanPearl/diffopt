try:
    from tqdm import auto as tqdm
except ImportError:
    tqdm = None

import jax.numpy as jnp
import scipy.optimize
from .adam import init_randkey

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
except ImportError:
    COMM = None


def trange_no_tqdm(n, desc=None, disable=False):
    return range(n)


def trange_with_tqdm(n, desc="BFGS Gradient Descent Progress", disable=False):
    return tqdm.trange(n, desc=desc, leave=True, disable=disable)


bfgs_trange = trange_no_tqdm if tqdm is None else trange_with_tqdm


def run_bfgs(loss_and_grad_fn, guess, maxsteps=100, param_bounds=None,
             randkey=None, thin=1, progress=True, comm=None):
    """Run the adam optimizer on a loss function with a custom gradient.

    Parameters
    ----------
    loss_and_grad_fn : callable
        Function with signature `loss_and_grad_fn(params) -> (loss, gradloss)`
    guess : array-like
        The starting parameters.
    maxsteps : int (default=100)
        The maximum number of steps to allowed.
    param_bounds : Sequence, optional
        Lower and upper bounds of each parameter of "shape" (ndim, 2). Pass
        `None` as the bound for each unbounded parameter, by default None
    randkey : int | PRNG Key (default=None)
        This will be passed to `logloss_and_grad_fn` under the "randkey" kwarg
    thin : int, optional
        Return parameters for every `thin` iterations, by default 1. Set
        `thin=0` to only return final parameters
    progress : bool, optional
        Display tqdm progress bar, by default True
    comm : MPI Communicator (default=COMM_WORLD)
        Communicator between all desired MPI ranks

    Returns
    -------
    params : jnp.array
        The trial parameters at each iteration.
    losses : jnp.array
        The loss values at each iteration.
    result : OptimizeResult (contains the following attributes):
        message : str, describes reason of termination
        success : boolean, True if converged
        fun : float, minimum loss found
        x : array of parameters at minimum loss found
        jac : array of gradient of loss at minimum loss found
        nfev : int, number of function evaluations
        nit : int, number of gradient descent iterations
    """
    kwargs = {}
    if randkey is not None:
        randkey = init_randkey(randkey)
        kwargs["randkey"] = randkey

    comm = comm if comm is not None else COMM
    if comm is None or comm.rank == 0:
        pbar = bfgs_trange(maxsteps, disable=not progress)
        params = []
        loss = []
        step = [-1]
        thindiv = thin if thin else maxsteps * len(params)

        # Wrap loss_and_grad function with commands to the worker ranks
        def loss_and_grad_fn_root(p):
            if comm is not None:
                comm.bcast("compute", root=0)
                comm.bcast(p)

            return loss_and_grad_fn(p, **kwargs)

        def callback(intermediate_result):
            if step[0] % thindiv == 0 or not len(params):
                params.append(intermediate_result.x)
                loss.append(intermediate_result.fun)
            else:
                params[-1] = intermediate_result.x
                loss[-1] = intermediate_result.fun
            step[0] += 1
            if hasattr(pbar, "update"):
                pbar.update()  # type: ignore

        result = scipy.optimize.minimize(
            loss_and_grad_fn_root, x0=guess, method="L-BFGS-B", jac=True,
            options=dict(maxiter=maxsteps), callback=callback,
            bounds=param_bounds)

        if not thin:
            params = params[-1]
            loss = loss[-1]
        if hasattr(pbar, "close"):
            pbar.close()  # type:ignore
        if comm is not None:
            comm.bcast("exit", root=0)
            comm.bcast([*result.keys()], root=0)
            comm.bcast([*result.values()], root=0)
            comm.bcast(params, root=0)
            comm.bcast(loss, root=0)

    else:
        while True:
            task = comm.bcast(None, root=0)

            if task == "compute":
                # receive params and execute loss function as ordered by root
                params = comm.bcast(None, root=0)
                loss_and_grad_fn(params, **kwargs)
            elif task == "exit":
                break
            else:
                raise ValueError("task %s not recognized!" % task)

        result_keys = comm.bcast(None, root=0)
        result_vals = comm.bcast(None, root=0)
        result = scipy.optimize.OptimizeResult(
            dict(zip(result_keys, result_vals)))
        params = comm.bcast(None, root=0)
        loss = comm.bcast(None, root=0)

    return jnp.array(params), jnp.array(loss), result
