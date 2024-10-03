import jax.numpy as jnp
import jax.random

from ... import kdescent


def test_kdescent():
    randkey = jax.random.key(1)
    training_x = jax.random.multivariate_normal(
        randkey, mean=jnp.array([1.0, 2.0, 3.0]),
        cov=jnp.array([[3.0, 0.2, -0.1],
                       [0.2, 5.0, 0.5],
                       [-0.1, 0.5, 7.0]]),
        shape=(100,))

    kde = kdescent.KCalc(training_x, num_kernels=10)

    @jax.jit
    def loss(params):
        model_x = training_x * 0.999 + params[None, :]
        model_kcounts, truth_kcounts = kde.compare_kde_counts(randkey, model_x)
        return jnp.mean((model_kcounts - truth_kcounts)**2)

    gradloss = jax.jit(jax.grad(loss))

    params1 = jnp.array([0., 0., 0.])
    params2 = jnp.array([0.9, -1.7, 2.4])
    params3 = jnp.array([1e20, -1e20, 1e20])

    assert loss(params3) > loss(params2) > loss(params1) > 0
    assert jnp.all(jnp.abs(gradloss(params1)) > 0)
    assert jnp.all(jnp.abs(gradloss(params2)) > 0)
    assert jnp.all(jnp.abs(gradloss(params3)) == 0)
