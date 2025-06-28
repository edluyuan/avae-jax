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
        self._key = random.PRNGKey(seed)

        # Initialize means and log-variances
        key, subkey = random.split(self._key)
        self._key = key
        self.locs = (random.uniform(subkey, (n_mixes, dim), minval=-0.5, maxval=0.5) * 2 * loc_scaling)
        log_var = jnp.ones((n_mixes, dim)) * log_var_scaling

        # Store scale_tril for diagonal covariance
        stds = jnp.exp(0.5 * log_var)
        self.scale_tril = jnp.stack([jnp.diag(stds[i]) for i in range(n_mixes)], axis=0)  # [K, D, D]

        # Uniform mixture weights
        self.log_weights = jnp.log(jnp.ones(n_mixes) / n_mixes)

    @jit
    def log_prob(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute log probability of x under the GMM.
        x: [..., D]
        returns: [...]
        """
        # Expand x to [..., 1, D] and locs to [1, K, D]
        x_exp = x[..., None, :]
        locs = self.locs[None, :, :]
        diffs = x_exp - locs  # [..., K, D]

        # Solve L y = diff for y, where scale_tril[k] = L
        # Compute for each mixture: mahalanobis squared
        def comp_mahal(tril, diff):
            # tril: [D, D], diff: [..., D]
            y = jax.scipy.linalg.solve_triangular(tril, diff.T, lower=True)
            return jnp.sum(y**2, axis=0)

        mahal = jnp.stack([comp_mahal(self.scale_tril[k], diffs[..., k, :])
                           for k in range(self.n_mixes)], axis=-1)  # [..., K]

        # Log determinants
        logdets = jnp.sum(jnp.log(jnp.diagonal(self.scale_tril, axis1=1, axis2=2)), axis=1)  # [K]
        const = -0.5 * (self.dim * jnp.log(2 * jnp.pi) + 2 * logdets)  # [K]

        # Component log probs: [..., K]
        comp_log = const + (-0.5 * mahal)
        # Add mixture log weights
        weighted = comp_log + self.log_weights
        # LogSumExp over mixtures
        return jax.scipy.special.logsumexp(weighted, axis=-1)

    def sample(self, shape: Tuple[int, ...], key: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Sample from the GMM.
        shape: output shape for samples (e.g. (N,)) will create samples of shape [..., D]
        """
        if key is None:
            key = self._key
        key, subkey = random.split(key)
        # Sample mixture indices
        mix_id = random.categorical(subkey, self.log_weights, shape=shape)
        # Sample standard normals
        key, subkey = random.split(key)
        eps = random.normal(subkey, shape + (self.dim,))
        # Gather locs and scale_tril
        locs = self.locs[mix_id]
        trils = self.scale_tril[mix_id]
        # Transform
        samples = jnp.einsum('...ij,...j->...i', trils, eps) + locs
        self._key = key
        return samples

    @property
    def test_set(self) -> jnp.ndarray:
        return self.sample((self.n_test_set_samples,))


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
