try:
    from tqdm import auto as tqdm
except ImportError:
    tqdm = None

import scipy.optimize
from .adam import init_randkey

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    N_RANKS = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    N_RANKS = 1


def trange_no_tqdm(n, desc=None):
    return range(n)


def trange_with_tqdm(n, desc="BFGS Gradient Descent Progress"):
    return tqdm.trange(n, desc=desc, leave=True)


bfgs_trange = trange_no_tqdm if tqdm is None else trange_with_tqdm


def run_bfgs(loss_and_grad_fn, params, maxsteps=100, param_bounds=None,
             randkey=None, comm=COMM):
    """Run the adam optimizer on a loss function with a custom gradient.

    Parameters
    ----------
    loss_and_grad_fn : callable
        Function with signature `loss_and_grad_fn(params) -> (loss, gradloss)`
    params : array-like
        The starting parameters.
    maxsteps : int (default=100)
        The maximum number of steps to allowed.
    param_bounds : Sequence, optional
        Lower and upper bounds of each parameter of "shape" (ndim, 2). Pass
        `None` as the bound for each unbounded parameter, by default None
    randkey : int | PRNG Key (default=None)
        This will be passed to `logloss_and_grad_fn` under the "randkey" kwarg
    comm : MPI Communicator (default=COMM_WORLD)
        Communicator between all desired MPI ranks

    Returns
    -------
    OptimizeResult (contains the following attributes):
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

    if comm is None or comm.rank == 0:
        pbar = bfgs_trange(maxsteps)

        # Wrap loss_and_grad function with commands to the worker ranks
        def loss_and_grad_fn_root(params):
            if comm is not None:
                comm.bcast("compute", root=0)
                comm.bcast(params)

            return loss_and_grad_fn(params, **kwargs)

        def callback(*_args, **_kwargs):
            if hasattr(pbar, "update"):
                pbar.update()  # type: ignore

        result = scipy.optimize.minimize(
            loss_and_grad_fn_root, x0=params, method="L-BFGS-B", jac=True,
            options=dict(maxiter=maxsteps), callback=callback,
            bounds=param_bounds)

        if hasattr(pbar, "close"):
            pbar.close()  # type:ignore
        if comm is not None:
            comm.bcast("exit", root=0)
            comm.bcast([*result.keys()], root=0)
            comm.bcast([*result.values()], root=0)

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

    return result
