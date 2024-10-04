"""mpiexec -n 3 python smf_grad_descent.py --num-halos 1_000_000
"""
import argparse
import time
from typing import NamedTuple
from dataclasses import dataclass

from mpi4py import MPI
import jax.scipy
from jax import numpy as jnp
import numpy as np

from .... import multigrad


# Optional: define a few NamedTuple datatypes for readability
class ParamTuple(NamedTuple):
    log_shmrat: float = -2.0
    sigma_logsm: float = 0.2


# Generate fake HMF as power law (truncated so that the SMF has a knee)
def load_halo_masses(num_halos=10_000, slope=-2, mmin=10.0 ** 10, qmax=0.95):
    q = jnp.linspace(0, qmax, num_halos)
    mhalo = mmin * (1 - q) ** (1/(slope+1))

    # Assign different halos to different MPI processes
    return np.array_split(mhalo, MPI.COMM_WORLD.size)[MPI.COMM_WORLD.rank]


# SMF helper functions
@jax.jit
def calc_smf_cdf(logsm, mean_logsm, sigma_logsm):
    return 0.5 * (1 + jax.scipy.special.erf(
        (logsm - mean_logsm)/(jnp.sqrt(2) * sigma_logsm)))


@jax.jit
def calc_smf_bin(params, logsm_low, logsm_high, volume, log_halo_masses):
    params = ParamTuple(*params)
    log_shmrat = params.log_shmrat
    sigma_logsm = params.sigma_logsm

    mean_logsm = log_halo_masses + log_shmrat

    cdf_high = calc_smf_cdf(logsm_high, mean_logsm, sigma_logsm)
    cdf_low = calc_smf_cdf(logsm_low, mean_logsm, sigma_logsm)
    return jnp.sum(cdf_high - cdf_low) / volume / (logsm_high - logsm_low)


# You must define a OnePointModel subclass, following this example:
@dataclass
class MySMFModel(multigrad.OnePointModel):
    # Optional: Update type hints and change default values as desired
    aux_data: dict
    loss_func_has_aux: bool = False
    # In addition to the aux_data attribute, note that you may
    # also directly use global values (works exactly the same)

    # You must define the following two differentiable + compilable methods
    # =====================================================================
    def calc_partial_sumstats_from_params(self, params):
        # This function should return an array of the PARTIAL sumstats
        # The TOTAL sumstats are obtained by summing over all MPI processes
        bin_edges = jnp.asarray(self.aux_data["smf_bin_edges"])
        log_halo_masses = jnp.asarray(self.aux_data["log_halo_masses"])
        volume = self.aux_data["volume"]

        smf = []
        logsm_low = bin_edges[0]
        for logsm_high in bin_edges[1:]:
            smf_bin = calc_smf_bin(
                params, logsm_low, logsm_high, volume, log_halo_masses)
            smf.append(smf_bin)
            logsm_low = logsm_high
        return jnp.array(smf)

    def calc_loss_from_sumstats(self, sumstats, sumstats_aux=None):
        target_sumstats = jnp.log10(self.aux_data["target_sumstats"])
        sumstats = jnp.log10(sumstats)
        # Reduced chi2 loss function assuming unit errors (mean squared error)
        return jnp.mean((sumstats - target_sumstats)**2)


parser = argparse.ArgumentParser(
    __file__, description="Example pipeline using multigrad to fit the SMF")
parser.add_argument("--num-halos", type=int, default=10_000)
parser.add_argument("--num-steps", type=int, default=2000)
parser.add_argument("--learning-rate", type=float, default=1e-3)

if __name__ == "__main__":
    args = parser.parse_args()
    data = dict(
        log_halo_masses=jnp.log10(load_halo_masses(args.num_halos)),
        smf_bin_edges=jnp.linspace(9, 10, 11),
        volume=10.0 * args.num_halos,  # Mpc^3/h^3
        target_sumstats=jnp.array([  # SMF at truth: params=(-2.0, 0.2)
            2.30178721e-02, 1.69728529e-02, 1.16054425e-02, 7.10532581e-03,
            3.77187086e-03, 1.69136131e-03, 6.28149020e-04, 1.90466686e-04,
            4.66692982e-05, 9.17260695e-06]),
    )
    model = MySMFModel(aux_data=data)

    guess = ParamTuple(log_shmrat=-1, sigma_logsm=0.5)
    t0 = time.time()
    gd_iterations = model.run_simple_grad_descent(
        guess=guess, nsteps=args.num_steps, learning_rate=args.learning_rate)
    gd_loss, gd_params = gd_iterations.loss, gd_iterations.params
    t = time.time() - t0

    # Parallel calculations needed for plots
    # ======================================
    truth = ParamTuple(log_shmrat=-2.0, sigma_logsm=0.2)
    final = ParamTuple(*gd_params[-1].tolist())
    guess_smf = model.calc_sumstats_from_params(guess)
    true_smf = model.calc_sumstats_from_params(truth)
    final_smf = model.calc_sumstats_from_params(final)

    # Report results and make plots on one process only
    # =================================================
    plot_results = True
    logmh_per_rank = MPI.COMM_WORLD.allgather(data["log_halo_masses"])
    if plot_results and not MPI.COMM_WORLD.Get_rank():
        import matplotlib.pyplot as plt

        print(f"Initial guess: {guess} ... {t} seconds later ...")
        print(f"Final solution: {final}")
        print(f"Truth: {truth}")
        print(f"True SMF: {repr(true_smf)}")

        # Plot the HMF
        logmh_min, logmh_max = logmh_per_rank[0][0], logmh_per_rank[-1][-1]
        bins = np.linspace(logmh_min, logmh_max, 101)
        for logmh in logmh_per_rank:
            plt.hist(logmh, bins=bins)
        plt.semilogy()
        plt.xlabel("$\\log M_h$", fontsize=16)
        plt.ylabel("$N$", fontsize=16)
        plt.savefig("hmf_model.png", bbox_inches="tight")
        plt.clf()

        # Plot the SMF target, initial guess, and final solution
        smf_bin_cens = 0.5 * (
            data["smf_bin_edges"][:-1] + data["smf_bin_edges"][1:])
        plt.semilogy(smf_bin_cens, true_smf, "go", label="Truth")
        plt.semilogy(
            smf_bin_cens, data["target_sumstats"], "rx", label="Target")
        plt.plot(smf_bin_cens, np.array(guess_smf),
                 "k--", label="Initial guess")
        plt.plot(smf_bin_cens, np.array(final_smf), label="Final solution")
        plt.xlabel("$\\log(M_\\star)$", fontsize=16)
        plt.ylabel(
            "$\\Phi(M_\\star)\\ [h^3{\\rm Mpc^{-3} dex^{-1}}]$", fontsize=16)
        plt.legend(frameon=False, fontsize=16)
        plt.savefig("smf_fit.png", bbox_inches="tight")
        plt.clf()

        # Plot the loss at each iteration of the grad-descent
        plt.plot(gd_loss)
        plt.semilogy()
        plt.xlabel("$N_{\\rm step}$", fontsize=16)
        plt.ylabel("$\\chi_\\nu^2$ loss", fontsize=16)
        plt.savefig("gd_loss.png", bbox_inches="tight")
        plt.clf()

        # Plot the params at each iteration of the grad-descent
        nrows = gd_params.shape[1]
        fig, axes = plt.subplots(nrows=nrows, figsize=(6.4, 4*nrows))
        for i in range(nrows):
            axes[i].plot(gd_params[:, i], label=ParamTuple._fields[i])
            axes[i].axhline(truth[i], color="r", ls="--", label="truth")
            if i == nrows - 1:
                axes[i].set_xlabel("$N_{\\rm step}$", fontsize=16)
            axes[i].set_ylabel(ParamTuple._fields[i], fontsize=16)
        plt.savefig("gd_param.png", bbox_inches="tight")
        plt.clf()

        # Plot a plot of the 2D path the parameters take
        plt.scatter(gd_params[:, 0], gd_params[:, 1], s=2)
        plt.plot(*truth, "rx", label="Truth")
        plt.xlabel(ParamTuple._fields[0], fontsize=16)
        plt.ylabel(ParamTuple._fields[1], fontsize=16)
        plt.legend(frameon=False, fontsize=16)
        plt.savefig("gd_param_path.png", bbox_inches="tight")
        plt.clf()
