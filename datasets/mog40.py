import jax
import jax.numpy as jnp
from jax import random, jit
import numpy as np
import matplotlib.pyplot as plt
import itertools
from typing import Tuple, Optional

class GMM:
    """
    2D Gaussian Mixture Model implemented in JAX.
    """
    def __init__(self,
                 dim: int,
                 n_mixes: int,
                 loc_scaling: float,
                 log_var_scaling: float = 0.1,
                 seed: int = 0,
                 n_test_set_samples: int = 1000):
        self.dim = dim
        self.n_mixes = n_mixes
        self.n_test_set_samples = n_test_set_samples
        # PRNG key

        self.key = random.key(seed)
        self.locs = (random.uniform(self.key, (n_mixes, dim), minval=-0.5, maxval=0.5) * 2 * loc_scaling)
        log_var = jnp.ones((n_mixes, dim)) * log_var_scaling

        # Store scale_tril for diagonal covariance
        stds = jnp.exp(0.5 * log_var)
        self.scale_tril = jnp.stack([jnp.diag(stds[i]) for i in range(n_mixes)], axis=0)  # [K, D, D]

        # Uniform mixture weights
        self.log_weights = jnp.log(jnp.ones(n_mixes) / n_mixes)

    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute log probability of x under the GMM.
        x: [..., D]
        returns: [...]
        """
        return _log_prob_jit(x, self.locs, self.scale_tril, self.log_weights, self.dim, self.n_mixes)

    def sample(self, shape: Tuple[int, ...], key: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Sample from the GMM.
        shape: output shape for samples (e.g. (N,)) will create samples of shape [..., D]
        """

        # Sample mixture indices
        mix_id = random.categorical(self.key, self.log_weights, shape=shape)
        # Sample standard normals

        eps = random.normal(self.key, shape + (self.dim,))
        # Gather locs and scale_tril
        locs = self.locs[mix_id]
        trils = self.scale_tril[mix_id]
        # Transform
        samples = jnp.einsum('...ij,...j->...i', trils, eps) + locs

        return samples

    @property
    def test_set(self) -> jnp.ndarray:
        return self.sample((self.n_test_set_samples,))


@jit
def _log_prob_jit(x, locs, scale_tril, log_weights, dim, n_mixes):
    """
    JIT-compiled log probability computation for the GMM.
    """
    # Expand x to [..., 1, D] and locs to [1, K, D]
    x_exp = x[..., None, :]
    locs_exp = locs[None, :, :]
    diffs = x_exp - locs_exp  # [..., K, D]

    # Vectorized computation of Mahalanobis distances
    # For each sample and mixture component, compute (x-mu)^T Sigma^{-1} (x-mu)
    # where Sigma^{-1} = (L L^T)^{-1} = L^{-T} L^{-1}

    # Solve L y = diff for each mixture component
    # diffs: [..., K, D], scale_tril: [K, D, D]
    # We need to solve scale_tril[k] @ y[..., k, :] = diffs[..., k, :] for each k

    def solve_triangular_batch(L, b):
        # L: [K, D, D], b: [..., K, D] -> [..., K, D]
        return jax.vmap(
            lambda L_k, b_k: jax.scipy.linalg.solve_triangular(L_k, b_k.T, lower=True).T,
            in_axes=(0, -2), out_axes=-2
        )(L, b)

    y = solve_triangular_batch(scale_tril, diffs)  # [..., K, D]
    mahal = jnp.sum(y**2, axis=-1)  # [..., K]

    # Log determinants
    logdets = jnp.sum(jnp.log(jnp.diagonal(scale_tril, axis1=1, axis2=2)), axis=1)  # [K]
    const = -0.5 * (dim * jnp.log(2 * jnp.pi) + 2 * logdets)  # [K]

    # Component log probs: [..., K]
    comp_log = const + (-0.5 * mahal)
    # Add mixture log weights
    weighted = comp_log + log_weights
    # LogSumExp over mixtures
    return jax.scipy.special.logsumexp(weighted, axis=-1)


def plot_contours(log_prob_func,
                  samples: Optional[jnp.ndarray] = None,
                  bounds: Tuple[float, float] = (-25.0, 25.0),
                  grid_width_n_points: int = 100,
                  n_contour_levels: Optional[int] = None,
                  log_prob_min: float = -1000.0,
                  plot_marginal_dims: Tuple[int, int] = (0, 1),
                  s: int = 2,
                  alpha: float = 0.6,
                  title: Optional[str] = None,
                  xy_tick: bool = True) -> None:
    x_vals = np.linspace(bounds[0], bounds[1], grid_width_n_points)
    grid = np.array(list(itertools.product(x_vals, x_vals)))  # [M, 2]
    # Evaluate log prob in JAX, convert to numpy
    logp = np.array(log_prob_func(jnp.array(grid)))
    logp = np.maximum(logp, log_prob_min)
    Z = logp.reshape((grid_width_n_points, grid_width_n_points))

    X = grid[:, 0].reshape((grid_width_n_points, grid_width_n_points))
    Y = grid[:, 1].reshape((grid_width_n_points, grid_width_n_points))

    plt.figure()
    if n_contour_levels:
        plt.contour(X, Y, Z, levels=n_contour_levels)
    else:
        plt.contour(X, Y, Z)
    if samples is not None:
        samples_np = np.clip(np.array(samples), bounds[0], bounds[1])
        plt.scatter(samples_np[:, plot_marginal_dims[0]],
                    samples_np[:, plot_marginal_dims[1]],
                    s=s, alpha=alpha)
        if xy_tick:
            plt.xticks([bounds[0], 0, bounds[1]])
            plt.yticks([bounds[0], 0, bounds[1]])
    if title:
        plt.title(title)
    plt.show()


def plot_MoG40(log_prob_function,
               samples: jnp.ndarray,
               file_name: Optional[str] = None,
               title: Optional[str] = None) -> None:
    if file_name:
        plt.ioff()
    plot_contours(log_prob_function,
                  samples=samples,
                  bounds=(-50, 50),
                  n_contour_levels=30,
                  grid_width_n_points=200,
                  title=title)
    if file_name:
        plt.savefig(file_name)
        plt.close()
