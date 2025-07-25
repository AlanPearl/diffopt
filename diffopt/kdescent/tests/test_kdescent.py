import unittest
import functools
import jax.numpy as jnp
import jax.random

from ... import kdescent


class TestKdescent(unittest.TestCase):
    def setUp(self):
        seed = 1
        randkey = jax.random.key(seed - 10)
        self.ndata = 100
        self.training_x = self.generate_data(randkey=randkey, ndata=self.ndata)
        pretrainer = kdescent.KPretrainer.from_training_data(
            self.training_x, num_eval_kernels=10,
            num_eval_fourier_positions=10,
            seed=seed)
        self.kde = kdescent.KCalc(pretrainer)
        self.randkey, = jax.random.split(randkey, 1)

    def generate_data(self, randkey, ndata=100):
        return jax.random.multivariate_normal(
            randkey, mean=jnp.array([1.0, 2.0, 3.0]),
            cov=jnp.array([[3.0, 0.2, -0.1],
                           [0.2, 5.0, 0.5],
                           [-0.1, 0.5, 7.0]]),
            shape=(ndata,))

    @functools.partial(jax.jit, static_argnames=["self", "rchisq"])
    def loss(self, params, randkey, rchisq=False):
        # Goes almost all the way down to loss=0 at params ~ [0, 0, 0]
        model_x = self.training_x * 0.999 + params[None, :]
        if rchisq:
            return self.kde.reduced_chisq_loss(randkey, model_x)
        else:
            model_kcounts, truth_kcounts = self.kde.compare_kde_counts(
                randkey, model_x, return_err=False)
            return jnp.mean((model_kcounts - truth_kcounts)**2)

    def test_loss_descent(self):
        gradloss = jax.jit(jax.grad(self.loss))

        params1 = jnp.array([0., 0., 0.])
        params2 = jnp.array([0.9, -1.7, 2.4])
        params3 = jnp.array([1e20, -1e20, 1e20])

        assert (self.loss(params3, self.randkey) >
                self.loss(params2, self.randkey) >
                self.loss(params1, self.randkey) > 0)
        assert (self.loss(params3, self.randkey, rchisq=True) >
                self.loss(params2, self.randkey, rchisq=True) >
                self.loss(params1, self.randkey, rchisq=True) > 0)
        assert jnp.all(jnp.abs(gradloss(params1, self.randkey)) > 0)
        assert jnp.all(jnp.abs(gradloss(params2, self.randkey)) > 0)
        assert jnp.all(jnp.abs(gradloss(params3, self.randkey)) == 0)

        params, losses = kdescent.adam(
            self.loss, params2, nsteps=100, progress=False,
            randkey=self.randkey, learning_rate=0.5)
        assert jnp.allclose(params[-1], params1, atol=0.1), "Misfit params"
        assert losses[0] / 1e3 > losses[-1], "Final loss must improve >1000x"
        params, losses = kdescent.adam(
            lambda *args, **kwargs: self.loss(*args, **kwargs, rchisq=True),
            params2, nsteps=100, progress=False,
            randkey=self.randkey, learning_rate=0.5)
        assert jnp.allclose(params[-1], params1, atol=0.1), "Misfit params"
        assert losses[0] / 1e3 > losses[-1], "Final loss must improve >1000x"

    def test_truth_uncertainty_chisq(self):
        # Use self.training_x and self.kde from setUp for the first sample
        test_x = self.generate_data(self.randkey, ndata=self.ndata)
        randkey, = jax.random.split(self.randkey, 1)

        # Compare KDE counts with uncertainty
        model_k, truth_k, err_k = self.kde.compare_kde_counts(
            randkey, test_x, return_err=True)

        model_f, truth_f, err_f = self.kde.compare_fourier_counts(
            randkey, test_x, return_err=True)

        normalized_residuals = jnp.concatenate([
            (model_k - truth_k) / err_k,
            (model_f.real - truth_f.real) / err_f.real,
            (model_f.imag - truth_f.imag) / err_f.imag
        ])

        reduced_chisq = jnp.mean(normalized_residuals**2)
        assert jnp.allclose(
            normalized_residuals, 0, atol=5
        ), "All residuals must be within 5 standard deviations"
        assert jnp.isclose(
            reduced_chisq, 2.5, atol=2.0
        ), f"Reduced chi^2 for KDE counts: {reduced_chisq}"
