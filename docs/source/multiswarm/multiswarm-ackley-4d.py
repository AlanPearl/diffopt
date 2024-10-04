"""
multiswarm-ackley-4d.py
"""

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from diffopt import multiswarm


def ackley(x_array):
    a = 20 * jnp.exp(-0.2 * jnp.sqrt(0.5 * (jnp.sum(x_array**2, axis=0))))
    b = jnp.exp(0.5 * jnp.sum(jnp.cos(2*jnp.pi*x_array), axis=0))
    return 20 + jnp.e - a - b


if __name__ == "__main__":
    swarm = multiswarm.ParticleSwarm(nparticles=100, ndim=4, xlow=-5, xhigh=5)
    results = swarm.run_pso(ackley, nsteps=100)

    best_loss, best_params = multiswarm.get_best_loss_and_params(
        results["swarm_loss_history"], results["swarm_x_history"])

    if not MPI.COMM_WORLD.rank:
        print("Best loss found =", best_loss)
        print("at params =", best_params)

        loss_histories = results["swarm_loss_history"]
        pos_histories = results["swarm_x_history"]
        best_loss_possible = ackley(np.zeros(pos_histories.shape[-1]))

        var_histories = np.var(pos_histories, axis=1)
        best_losses = np.min(loss_histories, axis=1)
        best_losses = np.minimum.accumulate(best_losses)
        fig, axes = plt.subplots(ncols=2, figsize=(11, 5))
        axes[0].plot(best_losses, color="C0")
        axes[0].axhline(best_loss_possible, color="k", ls="--")
        axes[1].semilogy(var_histories)
        axes[0].set_xlabel("t", fontsize=14)
        axes[0].set_ylabel("Best loss", fontsize=14)
        axes[1].set_xlabel("t", fontsize=14)
        axes[1].set_ylabel("var($x_i$) for i=1-4", fontsize=14)
        plt.savefig("ackley-fit-results.png")
