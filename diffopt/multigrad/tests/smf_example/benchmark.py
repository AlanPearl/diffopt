import argparse
from mpi4py import MPI

import jax
import jax.numpy as jnp

import smf_grad_descent as sgd


def speedtest(model, guess, nsteps=100, learning_rate=1e-3):
    model.run_grad_descent(
        guess=guess, nsteps=nsteps, learning_rate=learning_rate)


parser = argparse.ArgumentParser(
    __file__,
    description="Speed test multigrad with the smf_grad_descent.py pipeline."
)
parser.add_argument("--num-halos", type=int, default=10_000)
parser.add_argument("--num-steps", type=int, default=100)
parser.add_argument("--learning-rate", type=float, default=1e-3)
parser.add_argument("--save", type=str, default=None)

if __name__ == "__main__":
    jax.config.update("jax_platform_name", "cpu")
    args = parser.parse_args()
    data = dict(
        log_halo_masses=jnp.log10(sgd.load_halo_masses(args.num_halos)),
        smf_bin_edges=jnp.linspace(9, 10, 11),
        volume=10.0 * args.num_halos,  # Mpc^3/h^3
        target_sumstats=jnp.array([  # SMF at truth: params=(-2.0, 0.2)
            2.30178721e-02, 1.69728529e-02, 1.16054425e-02, 7.10532581e-03,
            3.77187086e-03, 1.69136131e-03, 6.28149020e-04, 1.90466686e-04,
            4.66692982e-05, 9.17260695e-06]),
    )
    model = sgd.MySMFModel(aux_data=data)
    guess = sgd.ParamTuple(log_shmrat=-1, sigma_logsm=0.5)
    nsteps = args.num_steps
    learning_rate = args.learning_rate

    # Run once to compile JIT methods
    speedtest(model, guess, 1, learning_rate)
    # Start the MPI timer and run the speed test
    t0 = MPI.Wtime()
    speedtest(model, guess, nsteps, learning_rate)
    t = MPI.Wtime() - t0

    if not MPI.COMM_WORLD.Get_rank():
        calls_per_sec = args.num_steps/t

        print(
            f"Benchmark with {MPI.COMM_WORLD.Get_size()} MPI processes {args}")
        print("=" * 70)
        print(f"Grad descent iterations/sec = {calls_per_sec}")
        print()

        if args.save is not None:
            result = dict(calls_per_sec=calls_per_sec,
                          num_processes=MPI.COMM_WORLD.Get_size(),
                          num_halos=args.num_halos,
                          num_steps=args.num_steps,
                          learning_rate=args.learning_rate,)
            nresults = 0
            try:
                with open(args.save) as f:
                    nresults = f.read().count("\n")
            except IOError:
                pass
            with open(args.save, "a+") as f:
                f.write(f"{repr(result)}\n")
