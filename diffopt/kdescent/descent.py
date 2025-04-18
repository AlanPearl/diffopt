from functools import partial

import scipy.optimize
import tqdm.auto as tqdm
import numpy as np
import jax.random
import jax.numpy as jnp
import jaxopt
import optax

from . import keygen


def adam(lossfunc, guess, nsteps=100, param_bounds=None,
         learning_rate=0.01, randkey=1, const_randkey=False,
         thin=1, progress=True, **other_kwargs):
    """
    Perform gradient descent

    Parameters
    ----------
    lossfunc : callable
        Function to be minimized via gradient descent. Must be compatible with
        jax.jit and jax.grad. Must have signature f(params, **other_kwargs)
    guess : array-like
            The starting parameters.
    nsteps : int, optional
        Number of gradient descent iterations to perform, by default 100
    param_bounds : Sequence, optional
        Lower and upper bounds of each parameter of "shape" (ndim, 2). Pass
        `None` as the bound for each unbounded parameter, by default None
    learning_rate : float, optional
        Initial Adam learning rate, by default 0.05
    randkey : int, optional
        Random seed or key, by default 1. If not None, lossfunc must accept
        the "randkey" keyword argument, e.g. `lossfunc(params, randkey=key)`
    const_randkey : bool, optional
        By default (False), randkey is regenerated at each gradient descent
        iteration. Remove this behavior by setting const_randkey=True
    thin : int, optional
        Return parameters for every `thin` iterations, by default 1. Set
        `thin=0` to only return final parameters
    progress : bool, optional
        Display tqdm progress bar, by default True

    Returns
    -------
    params : jnp.array
        List of params throughout the entire gradient descent, of shape
        (nsteps, n_param)
    """
    if param_bounds is None:
        return adam_unbounded(
            lossfunc, guess, nsteps, learning_rate, randkey,
            const_randkey, thin, progress, **other_kwargs)

    assert len(guess) == len(param_bounds)
    if hasattr(param_bounds, "tolist"):
        param_bounds = param_bounds.tolist()
    param_bounds = [b if b is None else tuple(b) for b in param_bounds]

    def ulossfunc(uparams, *args, **kwargs):
        params = apply_inverse_transforms(uparams, param_bounds)
        return lossfunc(params, *args, **kwargs)

    init_uparams = apply_transforms(guess, param_bounds)
    uparams, loss = adam_unbounded(
        ulossfunc, init_uparams, nsteps, learning_rate, randkey,
        const_randkey, thin, progress, **other_kwargs)
    params = apply_inverse_transforms(uparams.T, param_bounds).T

    return params, loss


def adam_unbounded(lossfunc, guess, nsteps=100, learning_rate=0.01,
                   randkey=1, const_randkey=False,
                   thin=1, progress=True, **other_kwargs):
    kwargs = {**other_kwargs}
    if randkey is not None:
        randkey = keygen.init_randkey(randkey)
        randkey, key_i = jax.random.split(randkey)
        kwargs["randkey"] = key_i
        if const_randkey:
            randkey = None
    opt = optax.adam(learning_rate)
    solver = jaxopt.OptaxSolver(opt=opt, fun=lossfunc, maxiter=nsteps)
    state = solver.init_state(guess, **kwargs)

    loss = []
    params = []
    params_i = guess
    thindiv = thin if thin else nsteps
    for i in tqdm.trange(nsteps + 1, disable=not progress,
                         desc="Adam Gradient Descent Progress"):
        if randkey is not None:
            randkey, key_i = jax.random.split(randkey)
            kwargs["randkey"] = key_i
        params_next, state = solver.update(params_i, state, **kwargs)
        if (i - 1) % thindiv == 0 or not len(params):
            loss.append(state.value)
            params.append(params_i)
        else:
            loss[-1] = state.value
            params[-1] = params_i
        if i < nsteps:
            params_i = params_next
    if not thin:
        params = params[-1]
        loss = loss[-1]

    return jnp.array(params), jnp.array(loss)


def bfgs(lossfunc, guess, maxsteps=100, param_bounds=None, randkey=None,
         thin=1, progress=True):
    """
    Run BFGS to descend the gradient and optimize the model parameters,
    given an initial guess. Stochasticity must be held fixed via a random key

    Parameters
    ----------
    lossfunc : callable
        Function to be minimized via gradient descent. Must be compatible with
        jax.jit and jax.grad. Must have signature f(params, **other_kwargs)
    guess : array-like
        The starting parameters.
    maxsteps : int, optional
        The maximum number of steps to take, by default 100.
    param_bounds : Sequence, optional
        Lower and upper bounds of each parameter of "shape" (ndim, 2). Pass
        `None` as the bound for each unbounded parameter, by default None
    randkey : int | PRNG Key, optional
        Since BFGS requires a deterministic function, this key will be
        passed to `calc_loss_and_grad_from_params()` as the "randkey" kwarg
        as a constant at every iteration, by default None
    thin : int, optional
        Return parameters for every `thin` iterations, by default 1. Set
        `thin=0` to only return final parameters
    progress : bool, optional
        Display tqdm progress bar, by default True

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
        randkey = keygen.init_randkey(randkey)
        kwargs["randkey"] = randkey

    pbar = tqdm.trange(maxsteps, desc="BFGS Gradient Descent Progress",
                       disable=not progress)
    params = []
    loss = []
    step = [-1]
    thindiv = thin if thin else maxsteps * len(guess)

    def callback(intermediate_result):
        if step[0] % thindiv == 0 or not len(params):
            params.append(intermediate_result.x)
            loss.append(intermediate_result.fun)
        else:
            params[-1] = intermediate_result.x
            loss[-1] = intermediate_result.fun
        step[0] += 1
        pbar.update()

    loss_and_grad_fn = jax.value_and_grad(
        lambda x: lossfunc(x, **kwargs))
    result = scipy.optimize.minimize(
        loss_and_grad_fn, x0=guess, method="L-BFGS-B", jac=True,
        options=dict(maxiter=maxsteps), callback=callback, bounds=param_bounds)
    if not thin:
        params = params[-1]
        loss = loss[-1]

    pbar.close()
    return jnp.array(params), jnp.array(loss), result


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
