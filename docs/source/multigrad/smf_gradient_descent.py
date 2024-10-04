from mpi4py import MPI
import jax.scipy
from jax import numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from diffopt import multigrad


def load_halo_masses(num_halos=10_000, comm=MPI.COMM_WORLD):
    # Generate fake halo masses between 10^10 < M_h < 10^11 as a power law
    quantile = jnp.linspace(0, 0.9, num_halos)
    mhalo = 1e10 / (1 - quantile)

    # Assign halos evenly across given MPI ranks (only one rank for now)
    return np.array_split(mhalo, comm.size)[comm.rank]


# Compute one bin of the stellar mass function (SMF)
@jax.jit
def calc_smf_bin(params, logsm_low, logsm_high, volume, log_halo_masses):
    # Unpack model parameters f and sigma
    log_f, log_sigma = params
    mean_logsm = log_f + log_halo_masses
    sigma = 10 ** log_sigma

    # Integrating the log-normal PDF over the bin, we get erf:
    erf_high = 0.5 * (1 + jax.scipy.special.erf(
        (logsm_high - mean_logsm)/(jnp.sqrt(2) * sigma)))
    erf_low = 0.5 * (1 + jax.scipy.special.erf(
        (logsm_low - mean_logsm)/(jnp.sqrt(2) * sigma)))
    prob_in_bin = erf_high - erf_low

    # Sum probabilities, convert to number density, divide by bin width
    return jnp.sum(prob_in_bin) / volume / (logsm_high - logsm_low)


# Compute the stellar mass function over all bins (loop over calc_smf_bin)
@jax.jit
def calc_smf(params, smf_bin_edges, volume, log_halo_masses):
    smf = []
    logsm_low = smf_bin_edges[0]
    for logsm_high in smf_bin_edges[1:]:
        smf_bin = calc_smf_bin(
            params, logsm_low, logsm_high, volume, log_halo_masses)
        smf.append(smf_bin)
        logsm_low = logsm_high
    return jnp.array(smf)


def plot_hmf_and_smf(smf, logmh_per_rank=None, plotarg="C0o-",
                     label=None, axes=None):
    if axes is None:
        _, axes = plt.subplots(ncols=2, figsize=(10.5, 4))
    if logmh_per_rank is not None:
        colors = [f"C{i}" for i in range(len(logmh_per_rank))]
        axes[0].hist(logmh_per_rank, bins=np.linspace(10, 11, 50),
                     color=colors, stacked=True)
        for i in range(len(colors)):
            axes[0].hist([], bins=np.linspace(10, 11, 50),
                         color=f"C{i}", label=f"MPI Rank = {i}")
        axes[0].legend(frameon=False)
        axes[0].semilogy()
        axes[0].set_xlabel("$\\log M_h$", fontsize=14)
        axes[0].set_ylabel("$\\Phi(\\log M_h)$", fontsize=14)

    smf_bin_cens = 0.5 * (smf_bin_edges[:-1] + smf_bin_edges[1:])
    axes[1].semilogy(smf_bin_cens, smf, plotarg, label=label)
    if label is not None:
        axes[1].legend(frameon=False)
    axes[1].set_xlabel("$\\log M_\\star$", fontsize=14)
    axes[1].set_ylabel("$\\Phi(\\log M_\\star)$", fontsize=14)
    return axes


class MySMFModel(multigrad.OnePointModel):
    def calc_partial_sumstats_from_params(self, params):
        # Accessing global variables is fine, but I prefer to store them in
        # the `aux_data` attribute, which we will define during construction
        bin_edges = self.aux_data["smf_bin_edges"]
        volume = self.aux_data["volume"]
        log_halo_masses = self.aux_data["log_halo_masses"]

        return calc_smf(params, bin_edges, volume, log_halo_masses)

    def calc_loss_from_sumstats(self, sumstats):
        # Add 1e-10 so that log values always remain finite
        target_sumstats = jnp.log10(self.aux_data["target_sumstats"] + 1e-10)
        sumstats = jnp.log10(sumstats + 1e-10)
        # Reduced chi2 loss function assuming unit errors (mean squared error)
        return jnp.mean((sumstats - target_sumstats)**2)


if __name__ == "__main__":
    volume = 1.0
    smf_bin_edges = jnp.linspace(9, 10, 11)
    true_params = jnp.array([-2.0, -0.5])

    log_halo_masses = jnp.log10(load_halo_masses(10_000))
    logmh_per_rank = MPI.COMM_WORLD.allgather(log_halo_masses)  # for plotting

    # We must sum calc_smf over all MPI ranks this time
    # Could equivalently use model.calc_sumstats_from_params(true_params)
    true_smf = multigrad.reduce_sum(
        calc_smf(true_params, smf_bin_edges, volume, log_halo_masses))

    aux_data = dict(
        log_halo_masses=log_halo_masses,
        smf_bin_edges=smf_bin_edges,
        volume=volume,
        target_sumstats=true_smf  # SMF at truth: params=(-2.0, -0.5)
    )
    model = MySMFModel(aux_data=aux_data)

    # Initial guess for our parameters. If it's too far off, there is always a
    # risk of getting stuck in local minima or other zero-valued gradients
    init_params = true_params + jnp.array([-1.5, 0.7])

    # Run gradient descent using the BFGS method powered by scipy
    results = model.run_bfgs(init_params)

    init_smf = model.calc_sumstats_from_params(init_params)
    final_smf = model.calc_sumstats_from_params(results.x)
    # Print and plot results from the root rank only
    if not MPI.COMM_WORLD.rank:
        print("BGFS has converged:", results.success)
        print("Initial guess =", init_params)
        print("True params =", true_params)
        print("Converged params =", results.x)
        print("\nFull results info:")
        print(results)

        axes = plot_hmf_and_smf(
            true_smf, logmh_per_rank, label="Truth")
        axes = plot_hmf_and_smf(
            init_smf, None, "k--", label="Initial guess", axes=axes)
        axes = plot_hmf_and_smf(
            final_smf, None, "r--", label="Best fit", axes=axes)

        plt.savefig("smf_gradient_descent.png", bbox_inches="tight")
