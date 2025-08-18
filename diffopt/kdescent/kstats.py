from functools import partial
from typing import overload, Tuple, Any, Literal
import dataclasses

import numpy as np
import jax.random
import jax.numpy as jnp


@dataclasses.dataclass
class KPretrainer:
    """
    Stores precomputed kernel and Fourier counts for training data,
    with kernel centers sampled from the training data PDF (via gaussian_kde).
    Provides save/load functionality.
    """
    kernel_centers: np.ndarray
    fourier_positions: np.ndarray
    kernelcov: np.ndarray
    kde_counts: np.ndarray
    kde_err: np.ndarray
    fourier_counts: np.ndarray
    fourier_err: np.ndarray
    num_eval_kernels: int
    num_eval_fourier_positions: int
    num_pretrain_kernels: int
    num_pretrain_fourier_positions: int
    bandwidth_factor: float
    fourier_range_factor: float
    covariant_kernels: bool
    inverse_density_weight_power: float
    training_sum_of_weights: float
    seed: int

    @classmethod
    def from_training_data(
        cls,
        training_x,
        training_weights=None,
        num_eval_kernels=None,
        num_eval_fourier_positions=None,
        num_pretrain_kernels=None,
        num_pretrain_fourier_positions=None,
        bandwidth_factor=1.0,
        fourier_range_factor=1.0,
        covariant_kernels=False,
        inverse_density_weight_power=0.0,
        num_idw_draws=None,
        chunk_size=None,
        seed=0,
        comm=None,
    ):
        """
        Create a pre-trained KPretrainer object from training data.

        Parameters
        ----------
        training_x : array-like
            Training data of shape (n_data, n_features)
        training_weights : array-like, optional
            Training weights of shape (n_data,), by default None
        num_eval_kernels : int, optional
            Number of KDE kernels to appriximate the PDF, by default 10*ndim
        num_eval_fourier_positions : int, optional
            Number of points to evaluate the ECF, by default 10*ndim
        num_pretrain_kernels : int, optional
            Number of KDE kernels to precompute training data PDF,
            by default 300*num_eval_kernels
        num_pretrain_fourier_positions : int, optional
            Number of points to precompute training data ECF,
            by default 300*num_eval_fourier_positions
        bandwidth_factor : float, optional
            Increase or decrease the kernel bandwidth, by default 1.0
        fourier_range_factor : float, optional
            Increase or decrease the Fourier search space, by default 1.0
        covariant_kernels : bool, optional
            If True, kernels will align with the principle
            components of the training data, which can blow up kernel count
            values if cov matrix has near-zero eigenvalues. By default False
        inverse_density_weight_power : float, optional
            At 1.0, this will weight the kernel selection by the inverse
            density of the training data. This is useful for selecting
            kernels in low-density regions. No selection weighting by default
        num_idw_draws : int, optional
            Number of KDE draws + evaluations in total for the importance
            resampling to determine kernel selection with inverse density
            weighting. By default 100*num_pretrain_kernels
        chunk_size : int, optional
            Chunk size for pre-computation of training KDE counts, to prevent
            memory overflow. If None, chunk_size will default to
            `max(5*num_eval_kernels, 5*num_eval_fourier_positions)`
        seed : int, optional
            Random seed for reproducibility, by default 0
        comm : MPI Communicator, optional
            Distribute pre-computation of training kernel counts across ranks,
            assuming full training data is loaded and identical across ranks.
        """
        seed = int(seed)
        randkeys = jax.random.split(jax.random.key(seed + 987), 3)
        training_x = jnp.atleast_2d(jnp.asarray(training_x).T).T
        assert training_x.ndim == 2, "x must have shape (ndata, ndim)"
        ndim = training_x.shape[1]
        # By default, use 10 * ndim evaluation kerels and fourier positions
        if num_eval_kernels is None:
            num_eval_kernels = 10 * ndim
        if num_eval_fourier_positions is None:
            num_eval_fourier_positions = 10 * ndim
        num_eval_kernels = int(num_eval_kernels)
        num_eval_fourier_positions = int(num_eval_fourier_positions)
        # By default, pretrain on 300 * number of evaluation kernels
        if num_pretrain_kernels is None:
            num_pretrain_kernels = 300 * num_eval_kernels
        if num_pretrain_fourier_positions is None:
            num_pretrain_fourier_positions = 300 * num_eval_fourier_positions
        if chunk_size is None:
            chunk_size = max(
                5 * num_eval_kernels, 5 * num_eval_fourier_positions, 1)
        num_pretrain_kernels = int(num_pretrain_kernels)
        num_pretrain_fourier_positions = int(num_pretrain_fourier_positions)
        chunk_size = int(chunk_size)

        # Bandwidth and kernel covariance
        bandwidth = _set_bandwidth(
            num_eval_kernels, ndim, bandwidth_factor)
        kernelcov = _bandwidth_to_kernelcov(
            training_x, bandwidth, training_weights, covariant_kernels
        )
        k_max = (fourier_range_factor / training_x.std(ddof=1, axis=0))

        # KDE for sampling kernel centers
        kde = jax.scipy.stats.gaussian_kde(
            training_x.T, weights=training_weights)

        # Importance resampling for inverse density weighting
        if inverse_density_weight_power > 0:
            if num_idw_draws is None:
                num_idw_draws = 100 * num_pretrain_kernels

            idw_chunk_size = chunk_size * 100
            num_chunks = num_idw_draws // idw_chunk_size + (
                num_idw_draws % idw_chunk_size > 0)
            pdf_vals = []

            draw_keys = jax.random.split(randkeys[2], num_chunks)
            draw_raw_samples = jax.jit(
                lambda x: kde.resample(x, (idw_chunk_size,)))
            compute_pdf_vals = jax.jit(lambda x: kde.pdf(x))
            for i in range(num_chunks):
                raw_samples = draw_raw_samples(draw_keys[i])
                pdf_vals.append(compute_pdf_vals(raw_samples))

            idw = jnp.concatenate(pdf_vals) ** (-inverse_density_weight_power)

            # Choose kernel centers with importance weights
            chosen_idx = jax.random.choice(
                randkeys[0], num_idw_draws, (num_pretrain_kernels,),
                p=idw[:num_idw_draws], replace=False)
            kernel_centers = jnp.asarray(raw_samples[:, chosen_idx].T)
        else:
            kernel_centers = jax.jit(
                lambda x: kde.resample(x, (num_pretrain_kernels,)))(
                    randkeys[2]).T

        # Sample fourier positions uniformly in k-space
        fourier_positions = jax.random.uniform(
            randkeys[1], (num_pretrain_fourier_positions, ndim)
        ) * k_max[None, :]

        if comm is not None:
            # Distribute kernel centers and fourier positions across ranks
            kernel_centers = np.array_split(
                kernel_centers, comm.size)[comm.rank]
            fourier_positions = np.array_split(
                fourier_positions, comm.size)[comm.rank]

        # Precompute KDE and Fourier counts for training data
        chunk_inds = list(range(
            chunk_size, len(kernel_centers), chunk_size))
        kde_counts, kde_err = np.concatenate([_predict_kde_counts(
            training_x, training_weights, x, kernelcov,
            return_err=True
        ) for x in np.array_split(kernel_centers, chunk_inds)], axis=1)
        chunk_inds = list(range(
            chunk_size, len(fourier_positions), chunk_size))
        fourier_counts, fourier_err = np.concatenate([_predict_fourier(
            training_x, training_weights, x, return_err=True
        ) for x in np.array_split(fourier_positions, chunk_inds)], axis=1)
        kernel_centers = np.asarray(kernel_centers)
        fourier_positions = np.asarray(fourier_positions)
        kde_counts = np.asarray(kde_counts)
        kde_err = np.asarray(kde_err)
        fourier_counts = np.asarray(fourier_counts)
        fourier_err = np.asarray(fourier_err)
        if comm is not None:
            # Gather all precomputed counts across all ranks
            kde_counts = np.concatenate(comm.allgather(kde_counts))
            kde_err = np.concatenate(comm.allgather(kde_err))
            fourier_counts = np.concatenate(comm.allgather(fourier_counts))
            fourier_err = np.concatenate(comm.allgather(fourier_err))
            kernel_centers = np.concatenate(comm.allgather(kernel_centers))
            fourier_positions = np.concatenate(
                comm.allgather(fourier_positions))

        training_sum_of_weights = len(training_x)
        if training_weights is not None:
            training_sum_of_weights = training_weights.sum()

        return cls(
            kernel_centers=kernel_centers,
            fourier_positions=fourier_positions,
            kernelcov=kernelcov,
            kde_counts=kde_counts,
            kde_err=kde_err,
            fourier_counts=fourier_counts,
            fourier_err=fourier_err,
            num_eval_kernels=num_eval_kernels,
            num_eval_fourier_positions=num_eval_fourier_positions,
            num_pretrain_kernels=num_pretrain_kernels,
            num_pretrain_fourier_positions=num_pretrain_fourier_positions,
            bandwidth_factor=bandwidth_factor,
            fourier_range_factor=fourier_range_factor,
            covariant_kernels=covariant_kernels,
            inverse_density_weight_power=inverse_density_weight_power,
            training_sum_of_weights=training_sum_of_weights,
            seed=seed
        )

    def save(self, filename):
        """Save the pre-trained object to disk as a .npz numpy zip file"""
        data = {field.name: getattr(self, field.name)
                for field in dataclasses.fields(self)}
        np.savez(filename, **data)

    @classmethod
    def load(cls, filename):
        """Load a pre-trained object from disk .npz file"""
        with np.load(filename, allow_pickle=False) as data:
            kwargs = {key: data[key] for key in data.files}
            return cls(**kwargs)


class KCalc:
    def __init__(self, pretrainer):
        """
        This KDE object is the fundamental building block of kdescent. It
        can be used to compare randomized evaluations of the PDF and ECF by
        training data to model predictions.

        Parameters
        ----------
        pretrainer : KPretrainer
            A pre-trained KPretrainer object that precomputes possible
            kernel centers and their associated training data counts.
        """
        if not isinstance(pretrainer, KPretrainer):
            raise TypeError("pretrainer must be an instance of KPretrainer")
        self.kernel_centers = jnp.array(pretrainer.kernel_centers)
        self.fourier_positions = jnp.array(pretrainer.fourier_positions)
        self.kernelcov = jnp.array(pretrainer.kernelcov)
        self.kde_counts = jnp.array(pretrainer.kde_counts)
        self.kde_err = jnp.array(pretrainer.kde_err)
        self.fourier_counts = jnp.array(pretrainer.fourier_counts)
        self.fourier_err = jnp.array(pretrainer.fourier_err)
        self.num_eval_kernels = int(pretrainer.num_eval_kernels)
        self.num_eval_fourier_positions = int(
            pretrainer.num_eval_fourier_positions)
        self.num_pretrain_kernels = int(pretrainer.num_pretrain_kernels)
        self.num_pretrain_fourier_positions = int(
            pretrainer.num_pretrain_fourier_positions)
        self.bandwidth_factor = float(pretrainer.bandwidth_factor)
        self.fourier_range_factor = float(pretrainer.fourier_range_factor)
        self.covariant_kernels = bool(pretrainer.covariant_kernels)
        self.inverse_density_weight_power = float(
            pretrainer.inverse_density_weight_power)
        self.training_sum_of_weights = float(
            pretrainer.training_sum_of_weights)

    def reduced_chisq_loss(self, randkey, x, weights=None, density=False):
        key1, key2 = jax.random.split(randkey, 2)
        model_k, truth_k, err_k = self.compare_kde_counts(
            key1, x, weights=weights, return_err=True)

        model_f, truth_f, err_f = self.compare_fourier_counts(
            key2, x, weights=weights, return_err=True)

        # Remove dependence of overall normalization if density=True
        if density:
            model_n = len(x)
            if weights is not None:
                model_n = weights.sum()
            model_k *= self.training_sum_of_weights / model_n
            model_f *= self.training_sum_of_weights / model_n

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
        return_err: Literal[False] = False, comm: Any = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        ...

    @overload
    def compare_kde_counts(
        self, randkey: Any, x: Any, weights: Any = None,
        return_err: Literal[True] = True, comm: Any = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        ...

    @overload
    def compare_fourier_counts(
        self, randkey: Any, x: Any, weights: Any = None,
        return_err: Literal[False] = False, comm: Any = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        ...

    @overload
    def compare_fourier_counts(
        self, randkey: Any, x: Any, weights: Any = None,
        return_err: Literal[True] = True, comm: Any = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        ...

    def compare_kde_counts(self, randkey, x, weights=None,
                           return_err=False, comm=None):
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
        comm : MPI Communicator, optional
            For parallel computing, this guarantees consistent kernel
            placements by all MPI ranks within the comm, by default None.
            WARNING: Do not pass in an MPI communicator here if you plan on
            JIT compiling; just pass identical randkeys for each MPI rank

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
        kernel_inds = self.realize_kernel_inds(randkey, comm)
        kernel_cens = self.kernel_centers[kernel_inds]
        prediction = _predict_kde_counts(
            x, weights, kernel_cens, self.kernelcov, return_err=False)
        truth = self.kde_counts[kernel_inds]
        if return_err:
            err = self.kde_err[kernel_inds]
            return prediction, truth, err
        else:
            return prediction, truth

    def compare_fourier_counts(self, randkey, x, weights=None,
                               return_err=False, comm=None):
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
        comm : MPI Communicator, optional
            For parallel computing, this guarantees consistent kernel
            placements by all MPI ranks within the comm, by default None.
            WARNING: Do not pass in an MPI communicator here if you plan on
            JIT compiling; just pass identical randkeys for each MPI rank

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
        fourier_inds = self.realize_fourier_inds(randkey, comm)
        fourier_positions = self.fourier_positions[fourier_inds]
        prediction = _predict_fourier(
            x, weights, fourier_positions, return_err=False)
        truth = self.fourier_counts[fourier_inds]
        if return_err:
            err = self.fourier_err[fourier_inds]
            return prediction, truth, err
        else:
            return prediction, truth

    def realize_kernel_inds(self, randkey, comm=None):
        if comm is None:
            return _sample_kernel_inds(
                self.num_eval_kernels, self.num_pretrain_kernels, randkey)
        else:
            kernel_inds = []
            if not comm.rank:
                kernel_inds = _sample_kernel_inds(
                    self.num_eval_kernels, self.num_pretrain_kernels, randkey)
            return comm.bcast(kernel_inds, root=0)

    def realize_fourier_inds(self, randkey, comm=None):
        if comm is None or comm.rank == 0:
            fourier_inds = _sample_kernel_inds(
                self.num_eval_fourier_positions,
                self.num_pretrain_fourier_positions, randkey)
            if comm is not None:
                comm.bcast(fourier_inds, root=0)
        else:
            fourier_inds = comm.bcast([], root=0)
        return fourier_inds


@jax.jit
def _set_bandwidth(n, d, bandwidth_factor):
    return n ** (-1.0 / (d + 4)) * bandwidth_factor


@partial(jax.jit, static_argnums=[3])
def _bandwidth_to_kernelcov(training_x, bandwidth, weights=None,
                            covariant_kernels=True):
    empirical_cov = jnp.cov(training_x, rowvar=False, aweights=weights)
    if not covariant_kernels:
        empirical_cov = jnp.diag(jnp.diag(empirical_cov))
    return empirical_cov * bandwidth**2


@partial(jax.jit, static_argnums=[0, 1])
def _sample_kernel_inds(num_samples, num_kernels, randkey):
    inds = jax.random.choice(
        randkey, num_kernels, (num_samples,), p=None)
    return inds


@jax.jit
def _weights_in_kernel(x, kernel_cen, cov):
    return jax.scipy.stats.multivariate_normal.pdf(
        x, mean=kernel_cen, cov=cov)


_vmap_weights_in_kernel = jax.jit(jax.vmap(
    _weights_in_kernel, in_axes=(None, 0, None)))


@jax.jit
def _get_kernel_probs(x, kernel_cens, cov):
    # ind_weights = [_weights_in_kernel(x, training_x, cov, ind)
    #                for ind in kernel_inds]
    ind_weights = _vmap_weights_in_kernel(x, kernel_cens, cov)
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
def _predict_kde_counts(x, x_weights, kernel_cens, cov, return_err=False):
    kernel_probs = _get_kernel_probs(x, kernel_cens, cov)
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
def _predict_fourier(x, x_weights, fourier_positions, return_err=False):
    exponentials = _get_fourier_exponentials(x, fourier_positions)
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
