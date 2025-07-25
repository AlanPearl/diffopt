import os
import unittest
try:
    from mpi4py.MPI import COMM_WORLD
    RANK = COMM_WORLD.RANK
except ImportError:
    COMM_WORLD = None
    RANK = 0

import jax.random
import jax.numpy as jnp

from ... import kdescent

SHOW_PLOTS = False


def barrier():
    if COMM_WORLD is None:
        return
    else:
        COMM_WORLD.COMM_WORLD.Barrier()


class TestPretrain(unittest.TestCase):
    def generate_data(self, randkey, ndata=100):
        return jax.random.multivariate_normal(
            randkey, mean=jnp.array([1.0, 2.0, 3.0]),
            cov=jnp.array([[3.0, 0.2, -0.1],
                           [0.2, 5.0, 0.5],
                           [-0.1, 0.5, 7.0]]),
            shape=(ndata,))

    def test_pretrain(self):
        pretrain_seed = 0
        randkey = jax.random.key(1)
        ndata = 10000
        training_x = self.generate_data(randkey=randkey, ndata=ndata)
        pretrain = kdescent.KPretrainer.from_training_data(
            training_x, num_pretrain_kernels=10,
            num_pretrain_fourier_positions=10,
            inverse_density_weight_power=0.5,
            seed=pretrain_seed, comm=COMM_WORLD)

        fn = "temp_kpretrain.npz"
        if not RANK:
            pretrain.save(fn)
        try:
            barrier()
            pretrain_loaded = kdescent.KPretrainer.load(fn)
        finally:
            barrier()  # Ensure all ranks finish before deleting
            if not RANK:
                os.remove(fn)

        # Check that the pretraining works
        assert pretrain_loaded.kernel_centers.shape == (10, 3)
        assert pretrain_loaded.kde_counts.shape == (10,)
        assert pretrain_loaded.kde_err.shape == (10,)
        assert pretrain_loaded.fourier_positions.shape == (10, 3)
        assert pretrain_loaded.fourier_counts.shape == (10,)
        assert pretrain_loaded.fourier_err.shape == (10,)

        # Check that the loaded pretrain matches the original
        assert jnp.all(pretrain_loaded.kernel_centers ==
                       pretrain.kernel_centers)
        assert jnp.all(pretrain_loaded.kde_counts == pretrain.kde_counts)
        assert jnp.all(pretrain_loaded.kde_err == pretrain.kde_err)

        if SHOW_PLOTS:
            import matplotlib.pyplot as plt
            plt.scatter(training_x[:, 0], training_x[:,
                        1], s=3, label="Training Data")
            plt.scatter(pretrain_loaded.kernel_centers[:, 0],
                        pretrain_loaded.kernel_centers[:, 1],
                        s=50, c='red', label="Kernel Centers")
            plt.legend(frameon=False, fontsize=14)
            plt.show()
