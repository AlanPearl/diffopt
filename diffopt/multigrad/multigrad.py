"""
"""
import math
from dataclasses import dataclass
from typing import Any, Union

import jax
import numpy as np
from jax import numpy as jnp

from . import util
from .adam import run_adam
from .bfgs import run_bfgs

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


def trange_no_tqdm(n, desc=None):
    return range(n)


def trange_with_tqdm(n, desc=None):
    return tqdm.trange(n, desc=desc)


trange = trange_no_tqdm if tqdm is None else trange_with_tqdm


def split_subcomms_by_node(comm=COMM):
    """
    Split comm into sub-comms (grouped by nodes)

    Parameters
    ----------
    comm : MPI.Comm, optional
        Specify a sub-communicator to split into sub-sub-communicators

    Returns
    -------
    subcomm: MPI.Comm
        The sub-comm that now controls this process
    num_groups: int
        The number of groups of subcomms (= number of nodes)
    group_rank: int
        The rank of this group (0 <= subcomm_rank < num_subcomms)
    """
    assert MPI is not None, "Cannot split subcomms without mpi4py"
    node_name = MPI.Get_processor_name()

    nodelist = comm.allgather(node_name)
    unique_nodelist = sorted(list(set(nodelist)))
    node_number = unique_nodelist.index(node_name)
    intra_node_id = len([i for i in nodelist[:RANK] if i == node_name])
    comm.Barrier()

    rankinfo = (RANK, intra_node_id, node_number)
    infolist = comm.allgather(rankinfo)
    sorted_infolist = sorted(infolist, key=lambda x: x[1])
    sorted_infolist = sorted(sorted_infolist, key=lambda x: x[2])

    subcomm = comm.Split(color=node_number)
    subcomm.Set_name(f"{comm.name}.{node_number}".replace(
        "MPI_COMM_WORLD.", ""))

    # subcomm.Free()  # Sometimes this cleanup helps prevent memory leaks
    return subcomm, len(unique_nodelist), node_number


def split_subcomms(num_groups=None, ranks_per_group=None,
                   comm=COMM):
    """
    Split comm into sub-comms (not grouped by nodes)

    Parameters
    ----------
    num_groups : int, optional
        Specify the number of evenly divided groups of subcomms
    ranks_per_group : list[int], optional
        Specify the number of ranks given to each sub-comm
    comm : MPI.Comm, optional
        Specify a sub-communicator to split into sub-sub-communicators

    Returns
    -------
    subcomm: MPI.Comm
        The sub-comm that now controls this process
    num_groups: int
        The number of groups of subcomms (same as input if not None)
    group_rank: int
        The rank of this group (0 <= subcomm_rank < num_subcomms)
    """
    assert comm is not None, "Cannot split subcomms without mpi4py"
    main_msg = "Specify either num_subcomms OR ranks_per_subcomm"
    sumrps_msg = "The sum of ranks_per_subcomm must equal comm.size"
    nsub_msg = "Cannot create more subcomms than there are ranks"
    if num_groups is not None:
        assert ranks_per_group is None, main_msg
        assert (comm.size >= num_groups), nsub_msg
        num_groups = int(num_groups)
        subnames = (np.ones(math.ceil(comm.size / num_groups))[None, :]
                    * np.arange(num_groups)[:, None])[:comm.size]
        subnames = subnames.ravel().astype(int)
    else:
        assert ranks_per_group is not None, main_msg
        assert sum(ranks_per_group) == comm.size, sumrps_msg
        num_groups = len(ranks_per_group)
        subnames = np.repeat(np.arange(num_groups), ranks_per_group)

    subname = str(np.array_split(subnames, comm.size)[comm.rank][0])

    nodelist = comm.allgather(subname)
    unique_nodelist = sorted(list(set(nodelist)))
    node_number = unique_nodelist.index(subname)
    intra_node_id = len([i for i in nodelist[:comm.rank] if i == subname])
    comm.Barrier()

    rankinfo = (comm.rank, intra_node_id, node_number)
    infolist = comm.allgather(rankinfo)
    sorted_infolist = sorted(infolist, key=lambda x: x[1])
    sorted_infolist = sorted(sorted_infolist, key=lambda x: x[2])

    sub_comm = comm.Split(color=node_number)
    sub_comm.Set_name(f"{comm.name}.{subname}".replace(
        "MPI_COMM_WORLD.", ""))

    # sub_comm.Free()  # Sometimes this cleanup helps prevent memory leaks
    return sub_comm, num_groups, int(subname)


def reduce_sum(value, root=None, comm=COMM):
    """Returns the sum of `value` across all MPI processes

    Parameters
    ----------
    value : np.ndarray | float | int
        value input by each MPI process to be summed
    root : int, optional
        rank of the process to receive and sum the values,
        by default None (broadcast result to all ranks)
    comm : MPI.Intracomm (default = MPI.COMM_WORLD)
        option to pass a sub-communicator in case the operation
        is not performed by all MPI ranks

    Returns
    -------
    np.ndarray | float
        Sum of values given by each rank of the communicator
    """
    if comm is None:
        return value
    return_to_scalar = not hasattr(value, "__len__")
    value = np.asarray(value)
    if root is None:
        # All-to-all sum
        total = np.empty_like(value)
        comm.Allreduce(value, total, op=MPI.SUM)
    else:
        # All-to-root sum
        total = np.empty_like(value)
        comm.Reduce(value, total, op=MPI.SUM, root=root)

    if return_to_scalar:
        total = total.tolist()
    return total


@dataclass
class OnePointModel:
    """
    Allows differentiable one-point calculations to be performed on separate
    MPI ranks, and automatically sums over each rank controlled by the comm.
    This is an abstract base class only. The user must personally define the
    `calc_partial_sumstats_from_params` and `calc_loss_from_sumstats` methods

    Parameters
    ----------
    aux_data : Any (default=None)
        Any auxiliary data for easy access within sumstats or loss functions
    comm : Comm (default=COMM_WORLD)
        MPI communicator
    loss_func_has_aux : bool (default=False)
        If true, `calc_partial_sumstats_from_params(x) -> (y, aux)` and
        `calc_loss_from_sumstats(y, aux) -> ...` signatures will be assumed
    sumstats_func_has_aux : bool (default=False)
        If true, `calc_loss_from_sumstats(...) -> (loss, aux)` signature
        will be assumed
    """
    aux_data: Any = None
    comm: Any = None
    loss_func_has_aux: bool = False
    sumstats_func_has_aux: bool = False

    def calc_partial_sumstats_from_params(self, params, randkey=None):
        """Custom method to map parameters to summary statistics"""
        raise NotImplementedError(
            "Subclass must implement `calc_partial_sumstats_func_from_params`"
        )

    def calc_loss_from_sumstats(self, sumstats, sumstats_aux=None,
                                randkey=None):
        """Custom method to map summary statistics to loss"""
        raise NotImplementedError(
            "Subclass must implement `calc_loss_func_from_sumstats`"
        )

    # NOTE: Never jit this method because it uses mpi4py
    def run_simple_grad_descent(self: Any, guess,
                                nsteps=100, learning_rate=0.01):
        """
        Descend the gradient with a fixed learning rate to optimize parameters,
        given an initial guess. Stochasticity not allowed.

        Parameters
        ----------
        guess : array-like
            The starting parameters.
        nsteps : int (default=100)
            The number of steps to take.
        learning_rate : float (default=0.001)
            The fixed learning rate.

        Returns
        -------
        GradientDescentResult (contains the following attributes):
            loss : array of loss values returned at each iteration
            params : array of trial parameters at each iteration
            aux : array of aux values returned at each iteration

        """
        return util.simple_grad_descent(
            None,
            guess=guess,
            nsteps=nsteps,
            learning_rate=learning_rate,
            loss_and_grad_func=self.calc_loss_and_grad_from_params,
            has_aux=False,
        )

    # NOTE: Never jit this method because it uses mpi4py
    def run_adam(self: Any, guess, nsteps=100, param_bounds=None,
                 learning_rate=0.01, randkey=None, const_randkey=False,
                 comm=None):
        """
        Run adam to descend the gradient and optimize the model parameters,
        given an initial guess. Stochasticity is allowed if randkey is passed.

        Parameters
        ----------
        guess : array-like
            The starting parameters.
        nsteps : int (default=100)
            The number of steps to take.
        param_bounds : Sequence, optional
            Lower and upper bounds of each parameter of "shape" (ndim, 2). Pass
            `None` as the bound for each unbounded parameter, by default None
        learning_rate : float (default=0.001)
            The adam learning rate.
        randkey : int | PRNG Key (default=None)
            If given, a new PRNG Key will be generated at each iteration and be
            passed to `calc_loss_and_grad_from_params()` as the "randkey" kwarg
        const_randkey : bool (default=False)
            By default, randkey is regenerated at each gradient descent
            iteration. Remove this behavior by setting const_randkey=True

        Returns
        -------
        array-like
            The optimal parameters.
        """
        comm = self.comm if comm is None else comm
        guess = jnp.asarray(guess)
        if const_randkey:
            def loss_and_grad_fn(x, _, **kw):
                return self.calc_loss_and_grad_from_params(
                    x, randkey=init_randkey, **kw)
            assert randkey is not None, "Must pass randkey if const_randkey"
            init_randkey = randkey
            randkey = None
        else:
            def loss_and_grad_fn(x, _, **kw):
                return self.calc_loss_and_grad_from_params(x, **kw)
        params_steps = run_adam(
            loss_and_grad_fn, params=guess, data=None, nsteps=nsteps,
            param_bounds=param_bounds, learning_rate=learning_rate,
            randkey=randkey
        )

        return jnp.asarray(comm.bcast(params_steps, root=0))

    # NOTE: Never jit this method because it uses mpi4py
    def run_bfgs(self: Any, guess, maxsteps=100, param_bounds=None,
                 randkey=None, comm=None):
        """
        Run BFGS to descend the gradient and optimize the model parameters,
        given an initial guess. Stochasticity must be held fixed via a random
        key

        Parameters
        ----------
        guess : array-like
            The starting parameters.
        maxsteps : int (default=100)
            The number of steps to take.
        param_bounds : Sequence, optional
            Lower and upper bounds of each parameter of "shape" (ndim, 2). Pass
            `None` as the bound for each unbounded parameter, by default None
        randkey : int | PRNG Key (default=None)
            Since BFGS requires a deterministic function, this key will be
            passed to `calc_loss_and_grad_from_params()` as the "randkey" kwarg
            as a constant at every iteration

        Returns
        -------
        OptimizeResult (contains the following attributes):
            message : str
                describes reason of termination
            success : boolean
                True if converged
            fun : float
                minimum loss found
            x : array
                parameters at minimum loss found
            jac : array
                gradient of loss at minimum loss found
            nfev : int
                number of function evaluations
            nit : int
                number of gradient descent iterations
        """
        comm = self.comm if comm is None else comm
        return run_bfgs(
            self.calc_loss_and_grad_from_params, guess, maxsteps=maxsteps,
            param_bounds=param_bounds, randkey=randkey, comm=comm)

    def run_lhs_param_scan(self, xmins, xmaxs, n_dim,
                           num_evaluations, seed=None, randkey=None):
        """
        Compute sumstat and loss values over a Latin Hypercube sample

        Parameters
        ----------
        xmins : float | array-like
            Lower bound on each parameter
        xmaxs : float | array-like
            Upper bound on each parameter
        n_dim : int
            Number of parameters
        num_evaluations : int
            Number of Latin Hypercube samples to draw and evaluate
        seed : int (default=None)
            Seed to make LHD draws reproducible, randomized by default
        randkey : PRNGKey | int (default=None)
            Random key passed to each sumstat and loss evaluation

        Returns
        -------
        params : array-like
            Parameters (drawn in Latin Hypercube shape)
        sumstats : array-like
            Sumstats evaluated at each draw of parameters
        losses : array-like
            Loss evaluated at each draw of parameters
        """
        params = util.latin_hypercube_sampler(xmins, xmaxs, n_dim,
                                              num_evaluations, seed=seed)
        rk = {} if randkey is None else {"randkey": randkey}
        sumstats = [self.calc_sumstats_from_params(x, **rk) for x in params]
        losses = [self.calc_loss_from_sumstats(x, **rk) for x in sumstats]
        return params, np.array(sumstats), np.array(losses)

    def __post_init__(self):
        if self.comm is None:
            self.comm = COMM
        # Create auto-diff functions needed for gradient descent
        self._grad_loss_from_sumstats = jax.grad(
            self.calc_loss_from_sumstats,
            has_aux=self.loss_func_has_aux)

    # sumstats functions
    # NOTE: Never jit this method because it uses mpi4py (when total=True)
    def calc_sumstats_from_params(
            self, params, total=True, randkey=None):
        """Compute summary statistics at given parameters

        Parameters
        ----------
        params : array-like
            Model parameters
        total : bool (default=True)
            If true (default), sumstats will be summed over all MPI ranks
        randkey : PRNGKey | int (default=None)
            If set to a value other than None, the "randkey" kwarg will be
            passed to user-defined methods

        Returns
        -------
        array
            Summary statistics evaluated at given parameters
        """
        kwargs = {} if randkey is None else {"randkey": randkey}
        result, aux = self.calc_partial_sumstats_from_params(
            params, **kwargs), None
        if self.sumstats_func_has_aux:
            result, aux = result
        if total:
            result = jnp.asarray(reduce_sum(result, comm=self.comm))
        result = (result, aux) if self.sumstats_func_has_aux else result
        return result

    # loss functions
    def calc_dloss_dsumstats(
            self, sumstats, sumstats_aux=None, randkey=None):
        kwargs = {} if randkey is None else {"randkey": randkey}
        sumstats = jnp.asarray(sumstats)
        args = (sumstats, sumstats_aux) if self.sumstats_func_has_aux else (
            sumstats,)
        return self._grad_loss_from_sumstats(*args, **kwargs)

    # NOTE: Never jit this method because it uses mpi4py
    def calc_loss_from_params(
            self, params, randkey=None):
        """Calculate the loss evaluated at a given set of parameters

        Parameters
        ----------
        params : array-like
            Model parameters
        randkey : PRNGKey | int (default=None)
            If set to a value other than None, the "randkey" kwarg will be
            passed to user-defined methods

        Returns
        -------
        float
            The loss evaluated at the parameters given
        """
        kwargs = {} if randkey is None else {"randkey": randkey}
        sumstats = self.calc_sumstats_from_params(params, **kwargs)
        if not self.sumstats_func_has_aux:
            sumstats = (sumstats,)
        return self.calc_loss_from_sumstats(*sumstats, **kwargs)

    # NOTE: Never jit this method because it uses mpi4py
    def calc_dloss_dparams(self, params, randkey=None):
        """Calculate the gradient of the loss w.r.t. model parameters given

        Parameters
        ----------
        params : array-like
            Model parameters
        randkey : PRNGKey | int (default=None)
            If set to a value other than None, the "randkey" kwarg will be
            passed to user-defined methods

        Returns
        -------
        array
            Gradient of the loss with respect to each parameter
        """
        return self._vjp(params, randkey=randkey, include_loss=False)

    # NOTE: Never jit this method because it uses mpi4py
    def calc_loss_and_grad_from_params(self, params, randkey=None):
        """
        Calculate the loss and its gradient.

        This function returns the equivalent of
        `(calc_loss_from_params(x), calc_dloss_dparams(x))` but it is
        significantly cheaper than calling them separately

        Parameters
        ----------
        params : array-like
            Model parameters
        randkey : PRNGKey | int (default=None)
            If set to a value other than None, the "randkey" kwarg will be
            passed to user-defined methods

        Returns
        -------
        float
            The loss evaluated at the parameters given
        array
            Gradient of the loss with respect to each parameter
        """
        return self._vjp(params, randkey=randkey, include_loss=True)

    # NOTE: Never jit this method because it uses mpi4py
    def _vjp(
        self, params, randkey=None, include_loss=True
    ):
        kwargs = {} if randkey is None else {"randkey": randkey}
        params = jnp.asarray(params)

        def sumstats_func(params):
            return self.calc_partial_sumstats_from_params(params, **kwargs)

        # Calculate sumstats AND save VJP func to perform chain rule later
        vjp_results = jax.vjp(
            sumstats_func, params,
            has_aux=self.sumstats_func_has_aux)  # type: ignore
        sumstats, vjp_func = vjp_results[:2]
        sumstats = jnp.asarray(reduce_sum(sumstats, comm=self.comm))
        args = (sumstats, *vjp_results[2:])

        # Calculate dloss_dsumstats for chain rule. Should be inexpensive
        dloss_dsumstats = self.calc_dloss_dsumstats(*args, **kwargs)
        if self.loss_func_has_aux:
            dloss_dsumstats = dloss_dsumstats[0]

        # Use VJP for the chain rule dL/dp[i] = sum(dL/ds[j] * ds[j]/dp[i])
        dloss_dparams = jnp.asarray(reduce_sum(
            vjp_func(dloss_dsumstats)[0], comm=self.comm))

        if include_loss:
            # Return (loss_and_aux, dloss_dparams)
            return self.calc_loss_from_sumstats(*args, **kwargs), dloss_dparams
        else:
            return dloss_dparams

    def __hash__(self):
        return hash((self.comm.name, self.calc_loss_from_sumstats))

    def __eq__(self, other):
        return isinstance(other, OnePointGroup) and self is other


@ dataclass
class OnePointGroup:
    """
    Allows different OnePointModels to simultaneously perform their
    calc_loss_and_grad_from_params method. The results are summed.

    Parameters
    ----------
    models : tuple[OnePointModel]
        Sequence of models, each providing a loss component to be summed.
    main_comm : Comm (default=COMM_WORLD)
        MPI communicator for the entire group (each model should be assigned
        its own sub-communicator)
    """
    models: Union[tuple[OnePointModel, ...], OnePointModel]
    main_comm: Any = None

    def __post_init__(self):
        if self.main_comm is None:
            self.main_comm = COMM
        if isinstance(self.models, OnePointModel):
            self.models = (self.models,)
        assert isinstance(self.models[0], OnePointModel)

    # NOTE: Never jit this method because it uses mpi4py
    def calc_loss_and_grad_from_params(self, params):
        loss, grad = [], []
        for model in self.models:
            res = model.calc_loss_and_grad_from_params(params)
            loss.append(res[0]*0 if model.comm.rank else res[0])
            grad.append(res[1]*0 if model.comm.rank else res[1])
        loss = jnp.concatenate(jnp.array(self.main_comm.allgather(loss)))
        grad = jnp.concatenate(jnp.array(self.main_comm.allgather(grad)))
        return loss.sum(), grad.sum(axis=0)

    # NOTE: Never jit this method because it uses mpi4py
    def run_simple_grad_descent(self, guess,
                                nsteps=100, learning_rate=0.01):
        return OnePointModel.run_simple_grad_descent(
            self, guess, nsteps, learning_rate)

    # NOTE: Never jit this method because it uses mpi4py
    def run_bfgs(self, guess, maxsteps=100, param_bounds=None, randkey=None):
        return OnePointModel.run_bfgs(
            self, guess, maxsteps, param_bounds=param_bounds,
            randkey=randkey, comm=self.main_comm)

    # NOTE: Never jit this method because it uses mpi4py
    def run_adam(self, guess, nsteps=100, param_bounds=None,
                 learning_rate=0.01, randkey=None, const_randkey=False):
        return OnePointModel.run_adam(
            self, guess, nsteps, param_bounds, learning_rate, randkey,
            const_randkey=const_randkey, comm=self.main_comm)

    def __hash__(self):
        if isinstance(self.models, OnePointModel):
            self.models = (self.models,)
        return hash((self.main_comm.name, self.models[0]))

    def __eq__(self, other):
        return isinstance(other, OnePointGroup) and self is other
