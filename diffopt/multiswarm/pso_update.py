"""Implementation of PSO algorithm described in arXiv:1108.5600 & arXiv:1310.7034"""  # noqa
from time import time

import numpy as np
import jax
from jax import numpy as jnp
from jax import random as jran
from scipy.stats import qmc

import tqdm.auto as tqdm

from .mpi_utils import split_subcomms

try:
    from mpi4py.MPI import COMM_WORLD
except ImportError:
    COMM_WORLD = None

# INERTIAL_WEIGHT = (0.5 / np.log(2))
# ACC_CONST = (0.5 + np.log(2))
INERTIAL_WEIGHT = 1.0
COGNITIVE_WEIGHT = 0.21
SOCIAL_WEIGHT = 0.07
VMAX_FRAC = 0.4


class ParticleSwarm:
    def __init__(self, nparticles, ndim, xlow, xhigh, seed=0,
                 inertial_weight=INERTIAL_WEIGHT,
                 cognitive_weight=COGNITIVE_WEIGHT,
                 social_weight=SOCIAL_WEIGHT,
                 vmax_frac=VMAX_FRAC,
                 ranks_per_particle=None,
                 comm=COMM_WORLD):
        """
        Initialize particles and MPI communicators to be used for PSO

        Parameters
        ----------
        nparticles : int
            Number of particles (~100+ recommended)
        ndim : int
            Dimensionality (i.e. number of model parameters to fit)
        xlow : int | Array[int]
            Lower bounds on each parameter
        xhigh : int | Array[int]
            Upper bounds on each parameter
        seed : int | PRNGKey, optional
            Seed for all pseudo-randomness, by default 0
        inertial_weight : float, optional
            Retain this fraction of the velocity from previous timestep, by
            default 1.0
        cognitive_weight : float, optional
            Weight pulling particles towards their personal best location ever
            found, by default 0.21
        social_weight : float, optional
            Weight pulling particles towards the global best location ever
            found, recommended ~1/3 of `cognitive_weight`, by default 0.07
        vmax_frac : float, optional
            Maximum velocity particles are allowed to travel, as a fraction of
            their box width per dimension, by default 0.4
        ranks_per_particle : int, optional
            Set this to manually control intra-particle parallelization, even
            if there are not enough ranks for nparticles * ranks_per_particle.
            By default (None), inter-particle parallelization is prioritized
        comm : MPI.Comm, optional
            MPI Communicator, by default COMM_WORLD
        """
        randkey = init_randkey(seed)
        subcomm, particles_on_this_rank = get_subcomm(
            nparticles, ranks_per_particle, comm=comm,
            return_particles_on_this_rank=True)

        num_particles_on_this_rank = len(particles_on_this_rank)
        init_key, *particle_keys = jran.split(
            randkey, nparticles + 1)
        particle_keys = [particle_keys[i] for i in particles_on_this_rank]
        init_cond = get_lhs_initial_conditions(
            nparticles, ndim, xlo=xlow, xhi=xhigh,
            vmax_frac=vmax_frac, ran_key=init_key)
        xmin, xmax, x_init, v_init = init_cond

        self.nparticles = nparticles
        self.ndim = ndim
        self.xlow, self.xhigh = xlow, xhigh
        self.comm = comm
        self.particles_on_this_rank = particles_on_this_rank
        self.num_particles_on_this_rank = num_particles_on_this_rank
        self.particle_keys = particle_keys
        self.subcomm = subcomm
        self.xmin, self.xmax = xmin, xmax
        self.x_init, self.v_init = x_init, v_init
        self.inertial_weight = inertial_weight
        self.cognitive_weight = cognitive_weight
        self.social_weight = social_weight
        self.vmax_frac = vmax_frac

    def run_pso(self, lossfunc, nsteps=100, keep_init_random_state=False):
        """
        Run particle swarm optimization (PSO)

        Parameters
        ----------
        lossfunc : callable
            The function we want to find the global minimum of. To be called
            with signature `lossfunc(x)` where x is an array of shape `(ndim,)`
        nsteps : int, optional
            Number of time step iterations, by default 100
        keep_init_random_state : bool, optional
            Set True to be able to rerun an identical run, or False (default)
            to continue a run by manually setting swarm.x_init and swarm.v_init

        Returns
        -------
        Results dictionary with the following keys:
            "swarm_x_history" : np.ndarray of shape (nsteps, nparticles, ndim)
                Position of all particles (trial params) at each time step
            "swarm_v_history": np.ndarray of shape (nsteps, nparticles, ndim)
                Velocity of all particles at each time step
            "swarm_loss_history": np.ndarray of shape (nsteps, nparticles)
                Loss of all particles at each time step
            "runtime": float
                Time in seconds, as measured on each rank, to perform PSO
        """
        if keep_init_random_state:
            particle_keys = self.particle_keys.copy()
        else:
            particle_keys = self.particle_keys

        x = [self.x_init[pr] for pr in self.particles_on_this_rank]
        v = [self.v_init[pr] for pr in self.particles_on_this_rank]

        loc_loss_best = [lossfunc(xi) for xi in x]
        loc_x_best = [np.copy(xi) for xi in x]

        swarm_x_best, swarm_loss_best = self._get_global_best(x, loc_loss_best)

        loc_x_history = [[] for _ in range(self.num_particles_on_this_rank)]
        loc_v_history = [[] for _ in range(self.num_particles_on_this_rank)]
        loc_loss_history = [[] for _ in range(self.num_particles_on_this_rank)]
        start = time()

        def trange(x):
            if self.comm.rank:
                return range(x)
            else:
                return tqdm.trange(x, desc="PSO Progress")
        for _ in trange(nsteps):
            istep_loss = [None for _ in range(self.num_particles_on_this_rank)]
            for ip in range(self.num_particles_on_this_rank):
                update_key = jran.split(particle_keys[ip], 1)[0]
                particle_keys[ip] = update_key
                x[ip], v[ip] = update_particle(
                    update_key, x[ip], v[ip], self.xmin, self.xmax,
                    loc_x_best[ip], swarm_x_best, self.inertial_weight,
                    self.cognitive_weight, self.social_weight, self.vmax_frac
                )
                istep_loss[ip] = lossfunc(x[ip])
            istep_x_best, istep_loss_best = self._get_global_best(
                x, istep_loss)

            for ip in range(self.num_particles_on_this_rank):
                if istep_loss_best <= swarm_loss_best:
                    swarm_loss_best = istep_loss_best
                    swarm_x_best = istep_x_best

                if istep_loss <= loc_loss_best:
                    loc_loss_best = istep_loss
                    loc_x_best = x

                loc_x_history[ip].append(x[ip])
                loc_v_history[ip].append(v[ip])
                loc_loss_history[ip].append(istep_loss[ip])

            # anneal = annealing_frac * self.inertial_weight
            # self.inertial_weight -= anneal
            # self.social_weight += anneal

        end = time()
        runtime = end - start

        if self.subcomm is not None and self.subcomm.rank > 0:
            # Only concatenate particles from the ROOT of each subcomm
            loc_x_history = np.zeros(shape=(0, *np.shape(loc_x_history[0])))
            loc_v_history = np.zeros(shape=(0, *np.shape(loc_v_history[0])))
            loc_loss_history = np.zeros(
                shape=(0, *np.shape(loc_loss_history[0])))

        swarm_x_history = np.concatenate(self.comm.allgather(
            loc_x_history), axis=0).swapaxes(0, 1)
        swarm_v_history = np.concatenate(self.comm.allgather(
            loc_v_history), axis=0).swapaxes(0, 1)
        swarm_loss_history = np.concatenate(self.comm.allgather(
            loc_loss_history), axis=0).swapaxes(0, 1)

        return {
            "swarm_x_history": swarm_x_history,
            "swarm_v_history": swarm_v_history,
            "swarm_loss_history": swarm_loss_history,
            "runtime": runtime
        }

    def _get_global_best(self, x, loss):
        if self.subcomm is not None and self.subcomm.rank > 0:
            # Only concatenate particles from the ROOT of each subcomm
            x = np.zeros(shape=(0, *np.shape(x[0])))
            loss = np.zeros(shape=(0, *np.shape(loss[0])))

        all_x = np.concatenate(self.comm.allgather(x))
        all_loss = np.concatenate(self.comm.allgather(loss))

        best_particle = np.argmin(all_loss)
        best_x = all_x[best_particle, :]
        best_loss = all_loss[best_particle]

        return best_x, best_loss


def get_subcomm(nparticles, ranks_per_particle=None, comm=COMM_WORLD,
                return_particles_on_this_rank=False):
    """
    Initialize MPI communicators to be used for PSO

    Parameters
    ----------
    nparticles : int
        Number of particles
    ranks_per_particle : int, optional
        Set this to manually control intra-particle parallelization, even
        if there are not enough ranks for nparticles * ranks_per_particle.
        By default (None), inter-particle parallelization is prioritized
    comm : MPI.Comm, optional
        MPI Communicator, by default COMM_WORLD
    return_particles_on_this_rank : bool, optional
        If true, return tuple (subcomm, particles_on_this_rank). By default,
        only subcomm is returned

    Returns
    -------
    subcomm : MPI.Comm
        This rank's subcommunicator, which can only talk to its "group"
    particles_on_this_rank : list
        If `return_particles_on_this_rank=True` this list will be returned,
        specifying the indices of particles this group is responsible for
    """
    assert comm is not None, "Please install mpi4py"
    rank, nranks = comm.Get_rank(), comm.Get_size()
    if ranks_per_particle is not None:
        # Set this to manually control intra-particle parallelization vs
        # inter-particle parallelization, even when there are not enough
        # ranks for nparticles * ranks_per_particle. By default,
        # inter-particle parallelization is prioritized.
        num_groups = comm.size / ranks_per_particle
        msg = "comm.size must be a multiple of ranks_per_particle"
        assert not num_groups % 1, msg
        num_groups = int(num_groups)
        subcomm, _, group_rank = split_subcomms(num_groups, comm=comm)
        particles_on_this_rank = np.array_split(
            np.arange(nparticles), num_groups)[group_rank].tolist()
    elif nparticles > nranks:
        particles_on_this_rank = np.array_split(
            np.arange(nparticles), nranks)[rank].tolist()
        subcomm = None
    else:
        subcomm, _, particles_on_this_rank = split_subcomms(
            nparticles, comm=comm)
        particles_on_this_rank = [particles_on_this_rank]

    if return_particles_on_this_rank:
        return subcomm, particles_on_this_rank
    else:
        return subcomm


def get_best_loss_and_params(loss_history, params_history):
    """
    Return the best loss and its corresponding parameters
    from the full results arrays returned by run_pso()

    Parameters
    ----------
    loss_history : Array[float] of shape (nsteps, nparticles)
        Loss of all particles at each time, given by "swarm_loss_history"
    params_history : Array[float] of shape (nsteps, nparticles, ndim)
        Position of all particles at each time, given by "swarm_x_history"

    Returns
    -------
    float
        Minimum loss value
    nd.ndarray[float]
        Parameters that produced the minimum loss
    """
    loss_history = np.ravel(loss_history)
    params_history = np.reshape(params_history, (*loss_history.shape, -1))

    best_arg = np.argmin(loss_history)
    best_loss = loss_history[best_arg]
    best_params = params_history[best_arg, :]
    return best_loss, best_params


def update_particle(
    ran_key,
    x,
    v,
    xmin,
    xmax,
    b_loc,
    b_swarm,
    w=INERTIAL_WEIGHT,
    acc_loc=COGNITIVE_WEIGHT,
    acc_swarm=SOCIAL_WEIGHT,
    vmax_frac=VMAX_FRAC
):
    xnew = x + v
    xnew, v = _impose_reflecting_boundary_condition(xnew, v, xmin, xmax)
    vnew = mc_update_velocity(
        ran_key, xnew, v, xmin, xmax, b_loc, b_swarm,
        w, acc_loc, acc_swarm, vmax_frac
    )
    return xnew, vnew


def mc_update_velocity(
    ran_key,
    x,
    v,
    xmin,
    xmax,
    b_loc,
    b_swarm,
    w=INERTIAL_WEIGHT,
    acc_loc=COGNITIVE_WEIGHT,
    acc_swarm=SOCIAL_WEIGHT,
    vmax_frac=VMAX_FRAC
):
    """Update the particle velocity

    Parameters
    ----------
    ran_key : jax.random.PRNGKey
        JAX random seed used to generate random speeds

    x : ndarray of shape (n_params, )
        Current position of particle

    xmin : ndarray of shape (n_params, )
        Minimum position of particle

    xmax : ndarray of shape (n_params, )
        Maximum position of particle

    v : ndarray of shape (n_params, )
        Current velocity of particle

    b_loc : ndarray of shape (n_params, )
        best point in history of particle

    b_swarm : ndarray of shape (n_params, )
        best point in history of swarm

    w : float, optional
        inertial weight
        Default is INERTIAL_WEIGHT defined at top of module

    acc_loc : float, optional
        local acceleration
        Default is ACC_CONST defined at top of module

    acc_swarm : float, optional
        swarm acceleration
        Default is ACC_CONST defined at top of module

    Returns
    -------
    vnew : ndarray of shape (n_params, )
        New velocity of particle

    """
    u_loc, u_swarm = jran.uniform(ran_key, shape=(2,))
    return _update_velocity_kern(
        x, v, xmin, xmax, b_loc, b_swarm, w, acc_loc,
        acc_swarm, vmax_frac, u_loc, u_swarm)


def _update_velocity_kern(
    x, v, xmin, xmax, b_loc, b_swarm, w, acc_loc,
    acc_swarm, vmax_frac, u_loc, u_swarm
):
    term1 = w * v
    term2 = u_loc * acc_loc * (b_loc - x)
    term3 = u_swarm * acc_swarm * (b_swarm - x)
    v = term1 + term2 + term3
    vmax = _get_vmax(xmin, xmax, vmax_frac)
    v = _get_clipped_velocity(v, vmax)
    # print(f"From x={x}: local_best={b_loc}, swarm_best={b_swarm}\n"
    #       f"v_inertia={term1}, v_cognitive={term2}, v_social={term3}",
    #       flush=True)
    return v


def _get_vmax(xmin, xmax, vmax_frac=VMAX_FRAC):
    return vmax_frac * (xmax - xmin)


def _get_clipped_velocity(v, vmax):
    # vmag = np.sqrt(np.sum(v**2))
    # if vmag > vmax:
    #     v = v * vmax / vmag
    v = np.where(v > vmax, vmax, v)
    v = np.where(v < -vmax, -vmax, v)
    return v


def _get_v_init(numpart, ran_key, xmin, xmax, vmax_frac=VMAX_FRAC):
    n_dim = xmin.size
    vmax = _get_vmax(xmin, xmax, vmax_frac)
    u_init = jran.uniform(ran_key, shape=(numpart, n_dim))
    return np.array(u_init * vmax)


def _impose_reflecting_boundary_condition(x, v, xmin, xmax):
    msk_lo = x < xmin
    msk_hi = x > xmax
    x = np.where(msk_lo, xmin, x)
    x = np.where(msk_hi, xmax, x)
    v = np.where(msk_lo | msk_hi, -v, v)
    return x, v


def get_lhs_initial_conditions(numpart, ndim, xlo=0, xhi=1, random_cd=True,
                               vmax_frac=VMAX_FRAC, ran_key=None):
    opt = "random-cd" if random_cd else None
    if ran_key is None:
        ran_key = jran.PRNGKey(987654321)
    xmin = np.zeros(ndim) + xlo
    xmax = np.zeros(ndim) + xhi
    x_init_key, v_init_key = jran.split(ran_key, 2)
    x_seed = int(jran.randint(
        x_init_key, (), 0, 1000000000, dtype=np.uint32))
    sampler = qmc.LatinHypercube(ndim, optimization=opt, seed=x_seed)
    x_init = sampler.random(numpart)
    x_init = qmc.scale(x_init, xmin, xmax)
    v_init = _get_v_init(numpart, v_init_key, xmin, xmax, vmax_frac)
    return xmin, xmax, x_init, v_init


def init_randkey(randkey) -> jax.Array:
    """Check that randkey is a PRNG key or create one from an int"""
    if isinstance(randkey, int):
        randkey = jran.key(randkey)
    else:
        msg = f"Invalid {type(randkey)=}: Must be int or PRNG Key"
        assert hasattr(randkey, "dtype"), msg
        assert jnp.issubdtype(randkey.dtype, jax.dtypes.prng_key), msg

    return randkey
