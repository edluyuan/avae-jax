import jax
import jax.numpy as jnp
from flax import linen as nn


class AffineVP(nn.Module):
    T: int
    alpha_0: float
    alpha_T: float

    def setup(self):
        alphas2 = jnp.linspace(self.alpha_0, self.alpha_T, self.T)
        sigmas = 1.0 - alphas2
        alpha2_cum = jnp.cumprod(alphas2)

        # Derived buffers
        self.alphas2 = alphas2           # [T]
        self.sigmas = sigmas             # [T]
        self.alpha2_cum = alpha2_cum     # [T]
        self.alpha_cum = jnp.sqrt(alpha2_cum)
        self.sqrt_1mac2 = jnp.sqrt(1.0 - alpha2_cum)

        # Precompute per-step coefficients
        c0 = []
        c1 = []
        sigma_cond = []
        inv_alpha = []
        sigma_ratio = []

        for t in range(self.T):
            t_next = jnp.minimum(t + 1, self.T - 1)

            ac_t2 = alpha2_cum[t]
            ac_t12 = alpha2_cum[t_next]
            s2_t = 1.0 - ac_t2
            s2_t1 = 1.0 - ac_t12
            sigma_t1 = sigmas[t_next]

            # c0[t] = alpha_cum[t] * sigma[t+1] / sqrt(1 - alpha2_cum[t+1])
            c0_val = (jnp.sqrt(ac_t2) * sigma_t1) / jnp.sqrt(s2_t1)
            # c1[t] = sqrt(alpha2[t+1]) * (1 - alpha2_cum[t]) / sqrt(1 - alpha2_cum[t+1])
            c1_val = (jnp.sqrt(alphas2[t_next]) * s2_t) / jnp.sqrt(s2_t1)
            # sigma_cond[t] = (s2_t * sigma[t+1]) / sqrt(1 - alpha2_cum[t+1])
            sigma_cond_val = (s2_t * sigma_t1) / jnp.sqrt(s2_t1)
            # inv_alpha[t] = 1/sqrt(alpha2[t+1])
            inv_alpha_val = 1.0 / jnp.sqrt(alphas2[t_next])
            # sigma_ratio[t] = sigma[t+1] / sqrt(1 - alpha2_cum[t+1])
            sigma_ratio_val = sigma_t1 / jnp.sqrt(s2_t1)

            c0.append(c0_val)
            c1.append(c1_val)
            sigma_cond.append(sigma_cond_val)
            inv_alpha.append(inv_alpha_val)
            sigma_ratio.append(sigma_ratio_val)

        # Stack into arrays of shape [T]
        self.c0 = jnp.stack(c0)
        self.c1 = jnp.stack(c1)
        self.sigma_cond = jnp.stack(sigma_cond)
        self.inv_alpha = jnp.stack(inv_alpha)
        self.sigma_ratio = jnp.stack(sigma_ratio)