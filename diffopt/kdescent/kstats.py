from functools import partial

import jax.random
import jax.numpy as jnp


class KCalc:
    def __init__(self, training_x, training_weights=None, num_kernels=20,
                 bandwidth_factor=0.4, num_fourier_kernels=20,
                 fourier_range_factor=4.0, covariant_kernels=True, comm=None):
        """
        This KDE object is the fundamental building block of kdescent. It
        can be used to compare randomized evaluations of the PDF and ECF by
        training data to model predictions.

        Parameters
        ----------
        training_x : array-like
            Training data of shape (n_data, n_features)
        training_weights : array-like, optional
            Training weights of shape (n_data,), by default None
        num_kernels : int, optional
            Number of KDE kernels to appriximate the PDF, by default 20
        bandwidth_factor : float, optional
            Increase or decrease the kernel bandwidth, by default 0.4
        num_fourier_kernels : int, optional
            Number of points in k-space to evaluate the ECF, by default 20
        fourier_range_factor : float, optional
            Increase or decrease the Fourier search space, by default 4.0
        covariant_kernels : bool, optional
            By default (True), kernels will align with the principle
            components of the training data, which can blow up kernel count
            values in nearly degenerate subspaces. Set False to prevent this
        comm : MPI Communicator, optional
            For parallel computing, this guarantees consistent kernel
            placements by all MPI ranks within the comm, by default None.
            WARNING: Do not pass in an MPI communicator if you plan on wrapping
            kernel drawing with a JIT-compiled function. In this case, be very
            careful to pass identical randkeys for each MPI rank
        """
        self.training_x = jnp.atleast_2d(jnp.asarray(training_x).T).T
        assert self.training_x.ndim == 2, "x must have shape (ndata, ndim)"
        self.training_weights = None
        if training_weights is not None:
            self.training_weights = jnp.asarray(training_weights)
            s = "training_weights must have shape (ndata,)"
            assert self.training_weights.shape == self.training_x.shape[:1], s
        self.comm = comm
        self.num_kernels = num_kernels
        self.ndim = self.training_x.shape[1]
        self.covariant_kernels = covariant_kernels
        self.bandwidth_factor = bandwidth_factor
        self.bandwidth = self._set_bandwidth(self.bandwidth_factor)
        self.kernelcov = self._bandwidth_to_kernelcov(self.bandwidth)
        self.num_fourier_kernels = num_fourier_kernels
        self.k_max = (fourier_range_factor
                      / self.training_x.std(ddof=1, axis=0))

    def compare_kde_counts(self, randkey, x, weights=None):
        """
        Realize kernel centers and return all kernel-weighted counts

        Parameters
        ----------
        x : array-like
            Model data of shape (n_model_data, n_features)
        weights : array-like, optional
            Effective counts with shape (n_model_data,). If supplied,
            function will return sum(weights * kernel_weights) within
            each kernel instead of simply sum(kernel_weights)

        Returns
        -------
        prediction : jnp.ndarray
            KDE counts measured on `x`. Has shape (num_kernels,)
        truth : jnp.ndarray
            KDE counts measured on `training_x`. This is always different
            due to the random kernel placements. Has shape (num_kernels,)
        """
        kde_kernels = self.realize_kde_kernels(randkey)
        prediction = self.calc_realized_kde(kde_kernels, x, weights)
        truth = self.calc_realized_training_kde(kde_kernels)
        return prediction, truth

    def compare_fourier_counts(self, randkey, x, weights=None):
        """
        Return randomly-placed evaluations of the ECF
        (Empirical Characteristic Function = Fourier-transformed PDF)

        Parameters
        ----------
        x : array-like
            Model data of shape (n_model_data, n_features)
        weights : array-like, optional
            Effective counts with shape (n_model_data,). If supplied,
            the ECF will be weighted as sum(weights * exp^(...)) at each
            evaluation in k-space instead of simply sum(exp^(...))

        Returns
        -------
        prediction : jnp.ndarray (complex-valued)
            CF evaluations measured on `x`. Has shape (num_kernels,)
        truth : jnp.ndarray (complex-valued)
            CF evaluations measured on `training_x`. This is always different
            due to the random evaluation kernels. Has shape (num_kernels,)
        """
        fourier_kernels = self.realize_fourier_kernels(randkey)
        prediction = self.calc_realized_fourier(fourier_kernels, x, weights)
        truth = self.calc_realized_training_fourier(fourier_kernels)
        return prediction, truth

    def realize_kde_kernels(self, randkey):
        if self.comm is None:
            return _sample_kernel_inds(
                self.num_kernels, self.training_x,
                self.training_weights, randkey)
        else:
            kernel_inds = []
            if not self.comm.rank:
                kernel_inds = _sample_kernel_inds(
                    self.num_kernels, self.training_x,
                    self.training_weights, randkey)
            return self.comm.bcast(kernel_inds, root=0)

    def realize_fourier_kernels(self, randkey):
        if self.comm is None:
            return _sample_fourier(
                self.num_fourier_kernels, self.k_max, randkey)
        else:
            k_kernels = []
            if not self.comm.rank:
                k_kernels = _sample_fourier(
                    self.num_fourier_kernels, self.k_max, randkey)
            return self.comm.bcast(k_kernels, root=0)

    def get_realized_weights(self, kernel_inds, x):
        return _get_weights(
            x, self.training_x, self.kernelcov, kernel_inds)

    def calc_realized_kde(self, kernel_inds, x, weights=None):
        return _predict_kdestat(
            x, weights, self.training_x, self.kernelcov, kernel_inds)

    def calc_realized_training_kde(self, kernel_inds):
        return self.calc_realized_kde(
            kernel_inds, self.training_x, self.training_weights)

    def calc_realized_fourier(self, fourier_kernels, x, weights=None):
        return _predict_fourier(x, weights, fourier_kernels)

    def calc_realized_training_fourier(self, fourier_kernels):
        return self.calc_realized_fourier(
            fourier_kernels, self.training_x, self.training_weights)

    def _set_bandwidth(self, bandwidth_factor):
        """Scott's rule bandwidth... multiplied by any factor you want!"""
        n = self.num_kernels
        d = self.training_x.shape[1]
        return _set_bandwidth(n, d, bandwidth_factor)

    def _bandwidth_to_kernelcov(self, bandwidth):
        """
        Scale bandwidth by the empirical covariance matrix. This way we
        don't have to perform a PC transform for every single iteration.
        """
        return _bandwidth_to_kernelcov(
            self.training_x, bandwidth, self.covariant_kernels)


@jax.jit
def _set_bandwidth(n, d, bandwidth_factor):
    return n ** (-1.0 / (d + 4)) * bandwidth_factor


@partial(jax.jit, static_argnums=[2])
def _bandwidth_to_kernelcov(training_x, bandwidth, covariant_kernels=True):
    empirical_cov = jnp.cov(training_x, rowvar=False)
    if not covariant_kernels:
        empirical_cov = jnp.diag(jnp.diag(empirical_cov))
    return empirical_cov * bandwidth**2


@partial(jax.jit, static_argnums=[0])
def _sample_kernel_inds(num_kernels, training_x, training_weights, randkey):
    inds = jax.random.choice(
        randkey, len(training_x), (num_kernels,), p=training_weights)
    return inds


@partial(jax.jit, static_argnums=[0])
def _sample_fourier(num_fourier_kernels, k_max, randkey):
    return jax.random.uniform(
        randkey, (num_fourier_kernels, len(k_max))
    ) * k_max[None, :]


@jax.jit
def _weights_in_kernel(x, training_x, cov, kernel_ind):
    x0 = training_x[kernel_ind, :]
    return jax.scipy.stats.multivariate_normal.pdf(
        x, mean=x0, cov=cov)


_vmap_weights_in_kernel = jax.jit(jax.vmap(
    _weights_in_kernel, in_axes=(None, None, None, 0)))


@jax.jit
def _get_weights(x, training_x, cov, kernel_inds):
    # ind_weights = [_weights_in_kernel(x, training_x, cov, ind)
    #                for ind in kernel_inds]
    ind_weights = _vmap_weights_in_kernel(x, training_x, cov, kernel_inds)
    return jnp.asarray(ind_weights)


@jax.jit
def _predict_kdestat_from_weights(x_weights, kernel_weights):
    if x_weights is None:
        return jnp.sum(kernel_weights, axis=1)
    else:
        return jnp.sum(x_weights[None, :] * kernel_weights, axis=1)


@jax.jit
def _predict_kdestat(x, x_weights, training_x, cov, kernel_inds):
    kernel_weights = _get_weights(x, training_x, cov, kernel_inds)
    return _predict_kdestat_from_weights(x_weights, kernel_weights)


@jax.jit
def _predict_fourier(x, x_weights, k_kernels):
    if x_weights is None:
        return jnp.sum(jnp.exp(
            1j * jnp.sum(k_kernels[:, None, :] * x[None, :, :], axis=-1)
        ), axis=-1)
    else:
        return jnp.sum(x_weights[None, :] * jnp.exp(
            1j * jnp.sum(k_kernels[:, None, :] * x[None, :, :], axis=-1)
        ), axis=-1)
