import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Tuple

from .nn import RevNet, GenNet, InfNet
from .scheduler import AffineVP


class Identity(nn.Module):
    z_dim: int = 2
    T: int = 1000
    alpha_0: float = 1 - 1e-6
    alpha_T: float = 0.96
    hidden: int = 64
    emb_dim: int = 32

    def setup(self):
        # precomputed schedule
        self.scheduler = AffineVP(self.T, self.alpha_0, self.alpha_T)
        # reverse network
        self._rev_network = RevNet(time_dim=self.emb_dim, hidden=self.hidden, z_dim=self.z_dim)
        # gamma_t: variance of reverse kernel
        # sigmas shape [T], we want [T, z_dim]
        self.gamma_t = jnp.expand_dims(self.scheduler.sigmas, 1) * jnp.ones((1, self.z_dim))

    def q(self, x0: jnp.ndarray, t: jnp.ndarray, eps: jnp.ndarray = None, key: jax.random.PRNGKey = None) -> jnp.ndarray:
        # forward kernel q(z_t | x0)
        if eps is None:
            assert key is not None, "Key must be provided when sampling eps"
            eps = jax.random.normal(key, x0.shape)
        alpha_cum = self.scheduler.alpha_cum[t]  # shape [batch]
        sqrt_1m_ac = self.scheduler.sqrt_1mac2[t]
        return alpha_cum[:, None] * x0 + sqrt_1m_ac[:, None] * eps

    def p_theta(self, z_t: jnp.ndarray, t: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # reverse kernel parameters
        c1_t = self.scheduler.c1[t]  # shape [batch]
        rho_t = self.scheduler.c0[t]
        affine = c1_t[:, None] * z_t
        mu_theta_prime = self._rev_network(z_t, t)
        res = rho_t[:, None] * mu_theta_prime
        mu_theta = affine + res
        gamma_t = self.gamma_t[t]
        return mu_theta, gamma_t, mu_theta_prime

    def f_theta_t(self,
                  z_tp1: jnp.ndarray,
                  x0: jnp.ndarray,
                  t: jnp.ndarray,
                  key: jax.random.PRNGKey,
                  antithetic_sampling: bool = True,
                  S: int = 12) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        A = self.scheduler.sigma_cond[t]
        rho_t = self.scheduler.c0[t]
        ac_t = jnp.expand_dims(self.scheduler.alpha_cum[t + 1], 1)
        mu_tilde = ac_t * x0
        sigma_tilde = jnp.sqrt(self.scheduler.s2_t[t + 1])
        m, d = z_tp1.shape
        # sample eps
        key_eps, _ = jax.random.split(key)
        eps = jax.random.normal(key_eps, (S, m, d))

        if not antithetic_sampling:
            z_s = mu_tilde[None, ...] + sigma_tilde[None, None] * eps
            z_rep = z_s.reshape((S * m, d))
            t_rep = jnp.repeat(t, S)
            _, _, mu_prime = self.p_theta(z_rep, t_rep)
            x0_rep = jnp.repeat(x0, S, axis=0)
            b_rep = (x0_rep - mu_prime) ** 2
            B_hat = (rho_t[:, None]**2) * b_rep.reshape((S, m, d)).mean(axis=0)
            f = A[:, None] + B_hat
        else:
            z_p = mu_tilde[None, ...] + sigma_tilde[None, None] * eps
            z_m = mu_tilde[None, ...] - sigma_tilde[None, None] * eps
            z_pm = jnp.concatenate([z_p, z_m], axis=0).reshape((2 * S * m, d))
            t_rep = jnp.repeat(t, 2 * S)
            _, _, mu_res_pm = self.p_theta(z_pm, t_rep)
            mu_res_p, mu_res_m = jnp.split(mu_res_pm, 2, axis=0)
            x0_rep = jnp.repeat(x0, S, axis=0)
            err_p = (x0_rep - mu_res_p) ** 2
            err_m = (x0_rep - mu_res_m) ** 2
            B_hat = rho_t[:, None]**2 * 0.5 * (err_p + err_m).reshape((S, m, d)).mean(axis=0)
            f = A[:, None] + B_hat
        return f, A, B_hat

    def sample(self, n: int, key: jax.random.PRNGKey) -> jnp.ndarray:
        # generate samples z0 or x
        key_z, key = jax.random.split(key)
        z_t = jax.random.normal(key_z, (n, self.z_dim))
        for t_idx in range(self.T - 1, 0, -1):
            t = jnp.full((n,), t_idx, dtype=jnp.int32)
            mu_theta_t, gamma_t, _ = self.p_theta(z_t, t)
            key_step, key = jax.random.split(key)
            eps = jax.random.normal(key_step, z_t.shape)
            z_t = mu_theta_t + jnp.sqrt(gamma_t)[:, None] * eps
        mu0, gamma0, _ = self.p_theta(z_t, jnp.zeros(n, dtype=jnp.int32))
        key0, _ = jax.random.split(key)
        eps0 = jax.random.normal(key0, mu0.shape)
        x_hat = mu0 + jnp.sqrt(gamma0)[:, None] * eps0
        return x_hat


class Latent(Identity):
    x_dim: int = 2

    def setup(self):
        super().setup()
        self._inf_network = InfNet(hidden=self.hidden, x_dim=self.x_dim, z_dim=self.z_dim)
        self._gen_network = GenNet(hidden=self.hidden, x_dim=self.x_dim, z_dim=self.z_dim)
        self.gamma_x = 0.01
        self.sigma0 = 0.01

    def q_phi(self, x0: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        mu0 = self._inf_network(x0)
        key_eps, _ = jax.random.split(key)
        eps = jax.random.normal(key_eps, mu0.shape)
        return mu0 + self.sigma0 * eps

    def p_chi(self, z0: jnp.ndarray) -> jnp.ndarray:
        return self._gen_network(z0)

    def sample(self, n: int, key: jax.random.PRNGKey) -> jnp.ndarray:
        key_z, key = jax.random.split(key)
        z_t = jax.random.normal(key_z, (n, self.z_dim))
        for t_idx in range(self.T - 1, 0, -1):
            t = jnp.full((n,), t_idx, dtype=jnp.int32)
            mu_theta_t, gamma_t, _ = self.p_theta(z_t, t)
            key_step, key = jax.random.split(key)
            eps = jax.random.normal(key_step, z_t.shape)
            z_t = mu_theta_t + jnp.sqrt(gamma_t)[:, None] * eps
        mu0, gamma0, _ = self.p_theta(z_t, jnp.zeros(n, dtype=jnp.int32))
        key0, key = jax.random.split(key)
        eps0 = jax.random.normal(key0, mu0.shape)
        z0 = mu0 + jnp.sqrt(gamma0)[:, None] * eps0
        x_mu = self.p_chi(z0)
        key_x, _ = jax.random.split(key)
        eps_x = jax.random.normal(key_x, x_mu.shape)
        return x_mu + jnp.sqrt(self.gamma_x) * eps_x


class Flow(Latent):
    # same as Latent but potentially different defaults or behavior
    pass
