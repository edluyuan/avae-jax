import jax.numpy as jnp
import flax.linen as nn

class SinusoidalPosEmb(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        half = self.dim // 2
        freqs = jnp.exp(-jnp.log(10000) * jnp.arange(half, dtype=jnp.float32) / (half - 1))
        angles = t.astype(jnp.float32)[:, None] * freqs[None, :]
        emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        return emb


