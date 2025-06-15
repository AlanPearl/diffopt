"""
Modified version of Matt's code taken from:
https://github.com/ArgonneCPAC/diff-ghmod-tools/blob/main/diff_ghmod_tools/adam.py  # noqa
"""
from functools import partial
try:
    from tqdm import auto as tqdm
except ImportError:
    tqdm = None

import numpy as np
import jax.random
import jax.numpy as jnp
from jax.example_libraries import optimizers as jax_opt

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
except ImportError:
    COMM = None


def trange_no_tqdm(n, desc=None, disable=False):
    return range(n)


def trange_with_tqdm(n, desc="Adam Gradient Descent Progress", disable=False):
    return tqdm.trange(n, desc=desc, disable=disable)


adam_trange = trange_no_tqdm if tqdm is None else trange_with_tqdm


def _master_wrapper(params, logloss_and_grad_fn, data, randkey=None,
                    comm=None):
    _comm = comm if comm is not None else COMM
    if _comm is not None:
        _comm.bcast("compute", root=0)
        _comm.bcast(params, root=0)

    kwargs = {}
    if randkey is not None:
        kwargs["randkey"] = randkey
    loss, grad = logloss_and_grad_fn(params, data, **kwargs)

    return loss, grad


def _adam_optimizer(params, fn, fn_data, nsteps, learning_rate, randkey=None,
                    thin=1, progress=True):
    kwargs = {}
    opt_init, opt_update, get_params = jax_opt.adam(learning_rate)
    opt_state = opt_init(params)

    param_steps = []
    loss_steps = []
    thindiv = thin if thin else nsteps
    for step in adam_trange(nsteps + 1, disable=not progress):
        if randkey is not None:
            randkey, key_i = jax.random.split(randkey)
            kwargs["randkey"] = key_i
        loss, grad = fn(params, *fn_data, **kwargs)
        if (step - 1) % thindiv == 0 or not len(param_steps):
            loss_steps.append(loss)
            param_steps.append(params)
        else:
            loss_steps[-1] = loss
            param_steps[-1] = params
        if step < nsteps:
            opt_state = opt_update(step, grad, opt_state)
            params = get_params(opt_state)
    if not thin:
        param_steps = param_steps[-1]
        loss_steps = loss_steps[-1]

    return jnp.array(param_steps), jnp.array(loss_steps)


def run_adam_unbounded(logloss_and_grad_fn, params, data, nsteps=100,
                       learning_rate=0.01, randkey=None,
                       thin=1, progress=True, comm=None):
    """Run the adam optimizer on a loss function with a custom gradient.

    Parameters
    ----------
    logloss_and_grad_fn : callable
        Function with signature logloss_and_grad_fn(params, data) that returns
        a 2-tuple of the loss and the gradient of the loss.
    params : array-like
        The starting parameters.
    data : anything
        The data passed to logloss_and_grad_fn
    nsteps : int
        The number of steps to take.
    learning_rate : float
        The adam learning rate.
    randkey : int | PRNG Key
        If given, a new PRNG Key will be generated at each iteration and be
        passed to `logloss_and_grad_fn` under the "randkey" kwarg
    thin : int, optional
        Return parameters for every `thin` iterations, by default 1. Set
        `thin=0` to only return final parameters
    progress : bool, optional
        Display tqdm progress bar, by default True
    comm : MPI.Comm, optional
        MPI communicator to use for parallelism

    Returns
    -------
    params : jnp.array
        The trial parameters at each iteration.
    losses : jnp.array
        The loss values at each iteration.
    """
    kwargs = {}
    if randkey is not None:
        randkey = init_randkey(randkey)
        kwargs["randkey"] = randkey

    _comm = comm if comm is not None else COMM
    _rank = 0 if _comm is None else _comm.rank

    if _rank == 0:
        fn = partial(_master_wrapper, comm=_comm)
        fn_data = (logloss_and_grad_fn, data)

        param_steps, loss_steps = _adam_optimizer(
            params, fn, fn_data, nsteps, learning_rate,
            randkey=randkey, thin=thin, progress=progress)

        if _comm is not None:
            _comm.bcast("exit", root=0)
    else:
        while True:
            task = _comm.bcast(None, root=0)

            if task == "compute":
                params = _comm.bcast(None, root=0)
                if randkey is not None:
                    randkey = gen_new_key(randkey)
                    kwargs["randkey"] = randkey
                logloss_and_grad_fn(params, data, **kwargs)
            elif task == "exit":
                break
            else:
                raise ValueError("task %s not recognized!" % task)

        param_steps = None
        loss_steps = None

    if _comm is not None:
        param_steps = _comm.bcast(param_steps, root=0)
        loss_steps = _comm.bcast(loss_steps, root=0)
    return jnp.asarray(param_steps), jnp.asarray(loss_steps)


def run_adam(logloss_and_grad_fn, params, data, nsteps=100, param_bounds=None,
             learning_rate=0.01, randkey=None, thin=1, progress=True,
             comm=None):
    """Run the adam optimizer on a loss function with a custom gradient.

    Parameters
    ----------
    logloss_and_grad_fn : callable
        Function with signature logloss_and_grad_fn(params, data) that returns
        a 2-tuple of the loss and the gradient of the loss.
    params : array-like
        The starting parameters.
    data : anything
        The data passed to logloss_and_grad_fn
    nsteps : int
        The number of steps to take.
    param_bounds : Sequence, optional
        Lower and upper bounds of each parameter of "shape" (ndim, 2). Pass
        `None` as the bound for each unbounded parameter, by default None
    learning_rate : float
        The adam learning rate.
    randkey : int | PRNG Key
        If given, a new PRNG Key will be generated at each iteration and be
        passed to `logloss_and_grad_fn` under the "randkey" kwarg
    thin : int, optional
        Return parameters for every `thin` iterations, by default 1. Set
        `thin=0` to only return final parameters
    progress : bool, optional
        Display tqdm progress bar, by default True
    comm : MPI.Comm, optional
        MPI communicator to use for parallelism

    Returns
    -------
    params : jnp.array
        The trial parameters at each iteration.
    losses : jnp.array
        The loss values at each iteration.
    """
    _comm = comm if comm is not None else COMM
    if param_bounds is None:
        return run_adam_unbounded(
            logloss_and_grad_fn, params, data, nsteps=nsteps,
            learning_rate=learning_rate, randkey=randkey,
            thin=thin, progress=progress, comm=_comm)

    assert len(params) == len(param_bounds)
    if hasattr(param_bounds, "tolist"):
        param_bounds = param_bounds.tolist()
    param_bounds = [b if b is None else tuple(b) for b in param_bounds]

    apply_trans = partial(apply_transforms, bounds=param_bounds)
    invert_trans = partial(apply_inverse_transforms, bounds=param_bounds)
    calc_dparams_duparams = jax.jacobian(invert_trans)

    def unbound_loss_and_grad(uparams, *args, **kwargs):
        params = invert_trans(uparams)
        loss, dloss_dparams = logloss_and_grad_fn(params, *args, **kwargs)
        dparams_duparams = calc_dparams_duparams(params)
        dloss_duparams = dloss_dparams @ dparams_duparams
        return loss, dloss_duparams

    uparams0 = apply_trans(params)
    uparams, loss = run_adam_unbounded(
        unbound_loss_and_grad, uparams0, data, nsteps, learning_rate, randkey,
        thin, progress, comm=_comm)

    params = invert_trans(uparams.T).T
    return params, loss


def apply_transforms(params, bounds):
    return jnp.array([transform(param, bound)
                      for param, bound in zip(params, bounds)])


def apply_inverse_transforms(uparams, bounds):
    return jnp.array([inverse_transform(uparam, bound)
                      for uparam, bound in zip(uparams, bounds)])


@partial(jax.jit, static_argnums=[1])
def transform(param, bounds):
    """Transform param into unbound param"""
    if bounds is None:
        return param
    low, high = bounds
    low_is_finite = low is not None and np.isfinite(low)
    high_is_finite = high is not None and np.isfinite(high)
    if low_is_finite and high_is_finite:
        mid = (high + low) / 2.0
        scale = (high - low) / jnp.pi
        return scale * jnp.tan((param - mid) / scale)
    elif low_is_finite:
        return param - low + 1.0 / (low - param)
    elif high_is_finite:
        return param - high + 1.0 / (high - param)
    else:
        return param


@partial(jax.jit, static_argnums=[1])
def inverse_transform(uparam, bounds):
    """Transform unbound param back into param"""
    if bounds is None:
        return uparam
    low, high = bounds
    low_is_finite = low is not None and np.isfinite(low)
    high_is_finite = high is not None and np.isfinite(high)
    if low_is_finite and high_is_finite:
        mid = (high + low) / 2.0
        scale = (high - low) / jnp.pi
        return mid + scale * jnp.arctan(uparam / scale)
    elif low_is_finite:
        return 0.5 * (2.0 * low + uparam + jnp.sqrt(uparam**2 + 4))
    elif high_is_finite:
        return 0.5 * (2.0 * high + uparam - jnp.sqrt(uparam**2 + 4))
    else:
        return uparam


def init_randkey(randkey):
    """Check that randkey is a PRNG key or create one from an int"""
    if isinstance(randkey, int):
        randkey = jax.random.key(randkey)
    else:
        msg = f"Invalid {type(randkey)=}: Must be int or PRNG Key"
        assert hasattr(randkey, "dtype"), msg
        assert jnp.issubdtype(randkey.dtype, jax.dtypes.prng_key), msg

    return randkey


@jax.jit
def gen_new_key(randkey):
    """Split PRNG key to generate a new one"""
    return jax.random.split(randkey, 1)[0]
