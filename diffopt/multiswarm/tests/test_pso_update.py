"""
"""
# import pytest
import numpy as np
from jax import random as jran
from .. import pso_update


TESTING_SEED = 43


def test_mc_update_velocity():
    n_dim = 2
    xmin = np.zeros(n_dim)
    xmax = np.ones(n_dim)

    ran_key = jran.PRNGKey(TESTING_SEED)
    pos_key, ran_key = jran.split(ran_key, 2)
    x = jran.uniform(pos_key, shape=(n_dim,), minval=xmin, maxval=xmax)
    v = np.zeros(n_dim) + 0.001
    b_loc = np.zeros(n_dim) + 0.5
    b_swarm = np.zeros(n_dim) + 0.5

    vnew = pso_update.mc_update_velocity(
        ran_key, x, v, xmin, xmax, b_loc, b_swarm)
    vmax = pso_update._get_vmax(xmin, xmax)
    assert np.all(vmax > 0)
    assert np.all(np.abs(vnew) <= vmax)


def test_get_v_init():
    n_dim = 2

    LO, HI = -100, 200
    ran_key = jran.PRNGKey(TESTING_SEED)
    n_tests = 50
    for itest in range(n_tests):
        v_init_key, xlo_key, xhi_key, ran_key = jran.split(ran_key, 4)
        xmin = jran.uniform(xlo_key, shape=(n_dim,), minval=LO, maxval=HI)
        delta = HI - xmin
        dx = jran.uniform(xlo_key, shape=(n_dim,), minval=0, maxval=delta)
        xmax = xmin + dx
        assert np.all(LO < xmin)
        assert np.all(xmin < xmax)
        assert np.all(xmax < HI)

        v_init = pso_update._get_v_init(1, v_init_key, xmin, xmax)
        vmax = pso_update._get_vmax(xmin, xmax)
        assert np.all(np.abs(v_init) < vmax)


def test_update_single_particle(seed=TESTING_SEED, n_updates=500):
    # TODO
    pass
