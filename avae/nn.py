import jax
import jax.numpy as jnp
from flax import linen as nn

from .emb import SinusoidalPosEmb


class RevNet(nn.Module):
    """
    Optimized reverse network: p_theta(z_t | z_tp1, t)
    """
    time_dim: int = 32
    hidden: int = 128
    z_dim: int = 2

    @nn.compact
    def __call__(self, z: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        # Time embedding - cached for efficiency
        t_emb = SinusoidalPosEmb(self.time_dim)(t)

        # Concatenate z and t_emb
        x = jnp.concatenate([z, t_emb], axis=-1)

        # Optimized feed-forward with residual connections
        # First block
        x_in = x
        x = nn.Dense(self.hidden, use_bias=False)(x)
        x = nn.LayerNorm()(x)
        x = nn.silu(x)

        # Second block with residual
        x_res = x
        x = nn.Dense(self.hidden, use_bias=False)(x)
        x = nn.LayerNorm()(x)
        x = nn.silu(x)
        x = x + x_res  # Residual connection

        # Third block with residual
        x_res = x
        x = nn.Dense(self.hidden, use_bias=False)(x)
        x = nn.LayerNorm()(x)
        x = nn.silu(x)
        x = x + x_res  # Residual connection

        # Final projection with skip connection from input
        x = nn.Dense(self.hidden, use_bias=False)(x)
        x = nn.LayerNorm()(x)
        x = nn.silu(x)

        # Project back to z_dim with zero initialization for better training
        x = nn.Dense(self.z_dim, kernel_init=nn.initializers.zeros)(x)

        return x


class GenNet(nn.Module):
    """
    Optimized generative network: p_chi(x|z0).
    """
    hidden: int = 128
    x_dim: int = 2
    z_dim: int = 2

    @nn.compact
    def __call__(self, z0: jnp.ndarray) -> jnp.ndarray:
        x = z0

        # Optimized architecture with residual connections
        x = nn.Dense(self.hidden, use_bias=False)(x)
        x = nn.LayerNorm()(x)
        x = nn.silu(x)

        # Residual blocks
        for _ in range(3):
            x_res = x
            x = nn.Dense(self.hidden, use_bias=False)(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)
            x = nn.Dense(self.hidden, use_bias=False)(x)
            x = nn.LayerNorm()(x)
            x = nn.silu(x)
            x = x + x_res  # Residual connection

        # Final projection
        x = nn.Dense(self.x_dim)(x)
        return x


class InfNet(nn.Module):
    """
    Optimized inference network: p_phi(z0|x)
    """
    hidden: int = 128
    x_dim: int = 2
    z_dim: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = x

        # Optimized architecture with residual connections
        h = nn.Dense(self.hidden, use_bias=False)(h)
        h = nn.LayerNorm()(h)
        h = nn.silu(h)

        # Residual blocks
        for _ in range(3):
            h_res = h
            h = nn.Dense(self.hidden, use_bias=False)(h)
            h = nn.LayerNorm()(h)
            h = nn.silu(h)
            h = nn.Dense(self.hidden, use_bias=False)(h)
            h = nn.LayerNorm()(h)
            h = nn.silu(h)
            h = h + h_res  # Residual connection

        # Final projection
        z = nn.Dense(self.z_dim)(h)
        return z