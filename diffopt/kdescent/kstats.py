from functools import partial
from typing import overload, Tuple, Any, Literal

import jax.random
import jax.numpy as jnp
from scipy import stats

IDW_SUBSAMPLE_SEED = 42
IDW_SUBSAMPLE_SIZE = 1000


class KCalc:
    def __init__(self, training_x, training_weights=None, num_kernels=20,
                 num_fourier_positions=20, bandwidth_factor=0.4,
                 fourier_range_factor=4.0, covariant_kernels=True,
                 inverse_density_weight_power=0.0, comm=None):
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
        num_fourier_positions : int, optional
            Number of points in k-space to evaluate the ECF, by default 20
        bandwidth_factor : float, optional
            Increase or decrease the kernel bandwidth, by default 0.4
        fourier_range_factor : float, optional
            Increase or decrease the Fourier search space, by default 4.0
        covariant_kernels : bool, optional
            By default (True), kernels will align with the principle
            components of the training data, which can blow up kernel count
            values in nearly degenerate subspaces. Set False to prevent this
        inverse_density_weight_power : float, optional
            At 1.0, this will weight the kernel selection by the inverse
            density of the training data. This is useful for selecting
            kernels in low-density regions. No selection weighting by default
        comm : MPI Communicator, optional
            For parallel computing, this guarantees consistent kernel
            placements by all MPI ranks within the comm, by default None.
            WARNING: Do not pass in an MPI communicator here if you plan on
            wrapping kernel drawing with a JIT-compiled function. In this case,
            just be very careful to pass identical randkeys for each MPI rank
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
        self.num_fourier_positions = num_fourier_positions
        self.k_max = (fourier_range_factor
                      / self.training_x.std(ddof=1, axis=0))

        self.idw_weights = None
        if inverse_density_weight_power > 0:
            key = jax.random.key(IDW_SUBSAMPLE_SEED)
            subsample = jax.random.randint(
                key, (IDW_SUBSAMPLE_SIZE,), 0, self.training_x.shape[0])
            subsample_x = self.training_x[subsample]
            subsample_w = self.training_weights[subsample] \
                if self.training_weights is not None else None
            training_density = stats.gaussian_kde(
                subsample_x.T, weights=subsample_w)(self.training_x.T)
            inv_weight = training_density ** inverse_density_weight_power
            if jnp.any(inv_weight == 0):
                inv_weight = inv_weight + inv_weight[inv_weight > 0].min()
            self.idw_weights = 1.0 / inv_weight

    def reduced_chisq_loss(self, randkey, x, weights=None, density=False):
        key1, key2 = jax.random.split(randkey, 2)
        model_k, truth_k, err_k = self.compare_kde_counts(
            key1, x, weights=weights, return_err=True)

        model_f, truth_f, err_f = self.compare_fourier_counts(
            key2, x, weights=weights, return_err=True)

        if density:
            # Remove dependence of overall normalization
            if weights is None:
                model_n = len(x)
            else:
                model_n = weights.sum()
            if self.training_weights is None:
                truth_n = len(self.training_x)
            else:
                truth_n = self.training_weights.sum()
            model_k *= truth_n / model_n
            model_f *= truth_n / model_n

        normalized_residuals = jnp.concatenate([
            (model_k - truth_k) / err_k,
            (model_f.real - truth_f.real) / err_f.real,
            (model_f.imag - truth_f.imag) / err_f.imag
        ])

        return jnp.mean(normalized_residuals**2)

    # Specify signatures to make linters happy
    @overload
    def compare_kde_counts(
        self, randkey: Any, x: Any, weights: Any = None,
        return_err: Literal[False] = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        ...

    @overload
    def compare_kde_counts(
        self, randkey: Any, x: Any, weights: Any = None,
        return_err: Literal[True] = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        ...

    @overload
    def compare_fourier_counts(
        self, randkey: Any, x: Any, weights: Any = None,
        return_err: Literal[False] = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        ...

    @overload
    def compare_fourier_counts(
        self, randkey: Any, x: Any, weights: Any = None,
        return_err: Literal[True] = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        ...

    def compare_kde_counts(self, randkey, x, weights=None, return_err=False):
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
        return_err: bool
            If true, also return the uncertainty of all training KDE counts
            values according to the effective sample size (ESS) in each kernel

        Returns
        -------
        prediction : jnp.ndarray
            KDE counts measured on `x`. Has shape (num_kernels,)
        truth : jnp.ndarray
            KDE counts measured on `training_x`. This is always different
            due to the random kernel placements. Has shape (num_kernels,)
        err : jnp.ndarray
            Returned if return_err=True, uncertainties of each KDE count
            in `truth` equal to truth/sqrt(ESS)
        """
        kde_kernels = self.realize_kde_kernels(randkey)
        prediction = self.calc_realized_kde(kde_kernels, x, weights)
        truth = self.calc_realized_training_kde(
            kde_kernels, return_err=return_err)
        if not return_err:
            return prediction, truth
        else:
            truth, err = truth
            return prediction, truth, err

    def compare_fourier_counts(self, randkey, x, weights=None,
                               return_err=False):
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
        return_err: bool
            If true, also return the uncertainty of all training Fourier counts
            values according to the effective sample size (ESS) in each kernel

        Returns
        -------
        prediction : jnp.ndarray (complex-valued)
            CF evaluations measured on `x`. Has shape (num_kernels,)
        truth : jnp.ndarray (complex-valued)
            CF evaluations measured on `training_x`. This is always different
            due to the random evaluation kernels. Has shape (num_kernels,)
        err : jnp.ndarray
            Returned if return_err=True, uncertainties of each Fourier count
            in `truth` equal to truth/sqrt(ESS)
        """
        fourier_positions = self.realize_fourier_positions(randkey)
        prediction = self.calc_realized_fourier(fourier_positions, x, weights)
        truth = self.calc_realized_training_fourier(
            fourier_positions, return_err=return_err)
        if not return_err:
            return prediction, truth
        else:
            truth, err = truth
            return prediction, truth, err

    def realize_kde_kernels(self, randkey):
        if self.comm is None:
            return _sample_kernel_inds(
                self.num_kernels, self.training_x,
                self.training_weights, self.idw_weights, randkey)
        else:
            kernel_inds = []
            if not self.comm.rank:
                kernel_inds = _sample_kernel_inds(
                    self.num_kernels, self.training_x,
                    self.training_weights, self.idw_weights, randkey)
            return self.comm.bcast(kernel_inds, root=0)

    def realize_fourier_positions(self, randkey):
        if self.comm is None:
            return _sample_fourier(
                self.num_fourier_positions, self.k_max, randkey)
        else:
            k_kernels = []
            if not self.comm.rank:
                k_kernels = _sample_fourier(
                    self.num_fourier_positions, self.k_max, randkey)
            return self.comm.bcast(k_kernels, root=0)

    def get_realized_kernel_probs(self, kernel_inds, x):
        return _get_kernel_probs(
            x, self.training_x, self.kernelcov, kernel_inds)

    def calc_realized_kde(self, kernel_inds, x, weights=None,
                          return_err=False):
        return _predict_kde_counts(
            x, weights, self.training_x, self.kernelcov, kernel_inds,
            return_err=return_err)

    def calc_realized_training_kde(self, kernel_inds, return_err=False):
        return self.calc_realized_kde(
            kernel_inds, self.training_x, self.training_weights,
            return_err=return_err)

    def calc_realized_fourier(self, fourier_positions, x, weights=None,
                              return_err=False):
        return _predict_fourier(
            x, weights, fourier_positions, return_err=return_err)

    def calc_realized_training_fourier(self, fourier_positions,
                                       return_err=False):
        return self.calc_realized_fourier(
            fourier_positions, self.training_x, self.training_weights,
            return_err=return_err)

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
def _sample_kernel_inds(num_kernels, training_x, training_weights,
                        idw_weights, randkey):
    choice_weights = idw_weights
    if idw_weights is None:
        choice_weights = training_weights
    elif training_weights is not None:
        choice_weights = idw_weights * training_weights

    inds = jax.random.choice(
        randkey, len(training_x), (num_kernels,), p=choice_weights)
    return inds


@partial(jax.jit, static_argnums=[0])
def _sample_fourier(num_fourier_positions, k_max, randkey):
    return jax.random.uniform(
        randkey, (num_fourier_positions, len(k_max))
    ) * k_max[None, :]


@jax.jit
def _weights_in_kernel(x, training_x, cov, kernel_ind):
    x0 = training_x[kernel_ind, :]
    return jax.scipy.stats.multivariate_normal.pdf(
        x, mean=x0, cov=cov)


_vmap_weights_in_kernel = jax.jit(jax.vmap(
    _weights_in_kernel, in_axes=(None, None, None, 0)))


@jax.jit
def _get_kernel_probs(x, training_x, cov, kernel_inds):
    # ind_weights = [_weights_in_kernel(x, training_x, cov, ind)
    #                for ind in kernel_inds]
    ind_weights = _vmap_weights_in_kernel(x, training_x, cov, kernel_inds)
    return jnp.asarray(ind_weights)


@jax.jit
def _get_fourier_exponentials(x, fourier_positions):
    return jnp.exp(
        1j * jnp.sum(fourier_positions[:, None, :] * x[None, :, :], axis=-1))


@jax.jit
def _weighted_sum_over_samples(kernel_probs, x_weights):
    if x_weights is None:
        return jnp.sum(kernel_probs, axis=1)
    else:
        return jnp.sum(x_weights[None, :] * kernel_probs, axis=1)


@partial(jax.jit, static_argnames=["return_err"])
def _predict_kde_counts(x, x_weights, training_x, cov, kernel_inds,
                        return_err=False):
    kernel_probs = _get_kernel_probs(x, training_x, cov, kernel_inds)
    kde_counts = _weighted_sum_over_samples(kernel_probs, x_weights)
    if return_err:
        x_weights_squared = None
        if x_weights is not None:
            x_weights_squared = x_weights ** 2
        ess = kde_counts ** 2 / _weighted_sum_over_samples(
            kernel_probs ** 2, x_weights_squared)
        err = kde_counts / jnp.sqrt(ess)
        return kde_counts, err
    else:
        return kde_counts


@partial(jax.jit, static_argnames=["return_err"])
def _predict_fourier(x, x_weights, k_kernels, return_err=False):
    exponentials = _get_fourier_exponentials(x, k_kernels)
    fourier_counts = _weighted_sum_over_samples(
        exponentials, x_weights)
    if return_err:
        x_weights_squared = None
        if x_weights is not None:
            x_weights_squared = x_weights ** 2
        ess_real = fourier_counts.real**2 / _weighted_sum_over_samples(
            exponentials.real**2, x_weights_squared)
        ess_imag = fourier_counts.imag**2 / _weighted_sum_over_samples(
            exponentials.imag**2, x_weights_squared)
        err_real = jnp.abs(fourier_counts.real) / jnp.sqrt(ess_real)
        err_imag = jnp.abs(fourier_counts.imag) / jnp.sqrt(ess_imag)
        return fourier_counts, err_real + 1j * err_imag
    else:
        return fourier_counts
