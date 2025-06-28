import jax
import jax.numpy as jnp
from flax import linen as nn


class AffineVP(nn.Module):
    T: int
    alpha_0: float
    alpha_T: float

    def setup(self):
        # Vectorized computation for better performance
        alphas2 = jnp.linspace(self.alpha_0, self.alpha_T, self.T)
        sigmas = 1.0 - alphas2
        alpha2_cum = jnp.cumprod(alphas2)

        # Derived buffers - all computed vectorized
        self.alphas2 = alphas2           # [T]
        self.sigmas = sigmas             # [T]
        self.alpha2_cum = alpha2_cum     # [T]
        self.alpha_cum = jnp.sqrt(alpha2_cum)
        self.sqrt_1mac2 = jnp.sqrt(1.0 - alpha2_cum)
        self.s2_t = 1.0 - alpha2_cum     # [T]

        # Vectorized computation of per-step coefficients
        t_indices = jnp.arange(self.T)
        t_next_indices = jnp.minimum(t_indices + 1, self.T - 1)

        ac_t2 = alpha2_cum[t_indices]
        ac_t12 = alpha2_cum[t_next_indices]
        s2_t = 1.0 - ac_t2
        s2_t1 = 1.0 - ac_t12
        sigma_t1 = sigmas[t_next_indices]

        # Vectorized coefficient computation
        self.c0 = (jnp.sqrt(ac_t2) * sigma_t1) / jnp.sqrt(s2_t1)
        self.c1 = (jnp.sqrt(alphas2[t_next_indices]) * s2_t) / jnp.sqrt(s2_t1)
        self.sigma_cond = (s2_t * sigma_t1) / jnp.sqrt(s2_t1)
        self.inv_alpha = 1.0 / jnp.sqrt(alphas2[t_next_indices])
        self.sigma_ratio = sigma_t1 / jnp.sqrt(s2_t1)
