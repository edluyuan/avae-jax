import jax
import jax.numpy as jnp
from flax import linen as nn

from .emb import SinusoidalPosEmb


class RevNet(nn.Module):
    """
    Reverse network: takes latent z and time t, outputs transformed z.
    """
    time_dim: int = 32
    hidden: int = 128
    z_dim: int = 2

    @nn.compact
    def __call__(self, z: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
        # Time embedding
        t_emb = SinusoidalPosEmb(self.time_dim)(t)
        # Concatenate z and t_emb
        x = jnp.concatenate([z, t_emb], axis=-1)
        # Feed-forward
        x = nn.Dense(self.hidden)(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden)(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden)(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden)(x)
        x = nn.silu(x)
        # Project back to z_dim
        x = nn.Dense(self.z_dim)(x)
        return x


class GenNet(nn.Module):
    """
    Generator network: maps initial latent z0 to data space x.
    """
    hidden: int = 128
    x_dim: int = 2
    z_dim: int = 2

    @nn.compact
    def __call__(self, z0: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.hidden)(z0)
        x = nn.silu(x)
        x = nn.Dense(self.hidden)(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden)(x)
        x = nn.silu(x)
        x = nn.Dense(self.hidden)(x)
        x = nn.silu(x)
        x = nn.Dense(self.x_dim)(x)
        return x


class InfNet(nn.Module):
    """
    Inference network: maps data x to latent z.
    """
    hidden: int = 128
    x_dim: int = 2
    z_dim: int = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        h = nn.Dense(self.hidden)(x)
        h = nn.silu(h)
        h = nn.Dense(self.hidden)(h)
        h = nn.silu(h)
        h = nn.Dense(self.hidden)(h)
        h = nn.silu(h)
        h = nn.Dense(self.hidden)(h)
        h = nn.silu(h)
        z = nn.Dense(self.z_dim)(h)
        return z