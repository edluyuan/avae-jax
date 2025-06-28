# train_identity_jax.py

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tqdm import trange
import matplotlib.pyplot as plt

from avae import Identity      # your Flax/Linen model
from datasets.mog40 import GMM, plot_MoG40        # your JAX GMM & plotting

# -----------------------------------------------------------------------------
# 1) Setup: seed, data, model, optimizer, initial schedules
# -----------------------------------------------------------------------------
rng = jax.random.PRNGKey(0)

# 1.1) 40‐component MoG dataset
mog40 = GMM(
    dim=2,
    n_mixes=40,
    loc_scaling=40.0,
    log_var_scaling=0.01,
    seed=0
)

# 1.2) Diffusion model
model = Identity(hidden=64, z_dim=2, emb_dim=32, T=1000)
rng, init_rng = jax.random.split(rng)
# dummy inputs to initialize
dummy_z = jnp.zeros((1, model.z_dim))
dummy_t = jnp.zeros((1,), dtype=jnp.int32)
# Flax init: returns a dict of variables, we only care about 'params'
variables = model.init(init_rng, dummy_z, dummy_t)
params = variables['params']

# 1.3) Optax Adam optimizer
tx = optax.adam(1e-3)
state = train_state.TrainState.create(apply_fn=model.apply,
                                      params=params,
                                      tx=tx)

# 1.4) Precompute initial w, q, and gamma_t (host‐side)
c0         = jnp.squeeze(model.scheduler.c0,        -1)   # [T]
sigma_cond = jnp.squeeze(model.scheduler.sigma_cond, -1)   # [T]
w_jax      = jnp.log1p(c0**2 * model.z_dim / sigma_cond)
q_jax      = w_jax / jnp.sum(w_jax)                     # discrete sampling weights
gamma_t_jax = jnp.repeat(model.scheduler.sigmas[:, None],
                         model.z_dim, axis=1)          # [T, z_dim]

# training hyperparameters
batch_size = 128
num_iters  = 1_000_000
S          = 12   # antithetic MC samples

# -----------------------------------------------------------------------------
# 2) jitted train_step
# -----------------------------------------------------------------------------
@jax.jit
def train_step(state, x0, q, rng):
    # split rng for t‐sampling and eps
    rng, rng_t, rng_eps = jax.random.split(rng, 3)

    # sample timesteps t ~ Categorical(q)
    logits = jnp.log(q)
    t = jax.random.categorical(rng_t, logits, shape=(x0.shape[0],))
    tp1 = jnp.minimum(t + 1, model.T - 1)

    # forward noising eps_{t+1}
    eps_tp1 = jax.random.normal(rng_eps, shape=x0.shape)

    def loss_fn(params):
        # q(z_{t+1}|x0)
        z_tp1 = state.apply_fn({'params': params},
                               x0, tp1, eps_tp1,
                               method=Identity.q)
        # energy f_t
        f, A, B_hat = state.apply_fn({'params': params},
                                     z_tp1, x0, t,
                                     antithetic_sampling=True, S=S,
                                     method=Identity.f_theta_t)
        # log‐energy & per‐batch loss
        log_e = jnp.log1p(B_hat / A)
        return 0.5 * jnp.mean(jnp.sum(log_e, axis=1) / q[t]), (f, log_e, t)

    # get grads + aux metrics
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (f, log_e, t_out)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, f, log_e, t_out, rng

# -----------------------------------------------------------------------------
# 3) Training loop (on host) with dynamic gamma_t & w updates
# -----------------------------------------------------------------------------
print("Starting training...")
for i in trange(num_iters):
    # 3.1) sample a batch from the GMM (as a NumPy array) and convert to jnp
    x0_batch = jnp.array(mog40.sample((batch_size,)))

    rng, step_rng = jax.random.split(rng)
    state, loss, f_val, log_e_val, t_idx, rng = train_step(
        state, x0_batch, q_jax, step_rng)

    # 3.2) host‐side update of gamma_t_jax and w_jax
    #     gather as NumPy for easy masking/averaging
    f_np       = np.array(f_val)
    log_e_np   = np.array(log_e_val)
    t_np       = np.array(t_idx)

    for ti in np.unique(t_np):
        mask     = (t_np == ti)
        gamma_new = f_np[mask].mean(axis=0)
        w_new     = log_e_np[mask].mean()

        # update the JAX arrays
        gamma_t_jax = gamma_t_jax.at[ti].set(gamma_new)
        w_jax       = w_jax.at[ti].set(w_new)

    # renormalize
    q_jax = w_jax / jnp.sum(w_jax)

    if (i + 1) % 100 == 0:
        print(f"Iter {i+1:>7}, loss {loss:.4f}")

# -----------------------------------------------------------------------------
# 4) JIT-compiled sampling via lax.scan
# -----------------------------------------------------------------------------
@jax.jit
def sample_fn(params, rng, n):
    # start from Gaussian at t=T
    rng, zrng = jax.random.split(rng)
    z = jax.random.normal(zrng, (n, model.z_dim))

    def step(carry, t):
        z_t, rng = carry
        rng, srng = jax.random.split(rng)
        # p_theta one step
        mu, gamma, _ = model.apply({'params': params},
                                   z_t, jnp.array([t]*n),
                                   method=Identity.p_theta)
        eps = jax.random.normal(srng, z_t.shape)
        z_next = mu + jnp.sqrt(gamma) * eps
        return (z_next, rng), None

    timesteps = jnp.arange(model.T - 1, 0, -1)
    (z_final, _), _ = jax.lax.scan(step, (z, rng), timesteps)

    # final reverse step t=0
    mu0, gamma0, _ = model.apply({'params': params},
                                  z_final,
                                  jnp.zeros((n,), dtype=jnp.int32),
                                  method=Identity.p_theta)
    rng, rng0 = jax.random.split(rng)
    eps0 = jax.random.normal(rng0, mu0.shape)
    x_hat = mu0 + jnp.sqrt(gamma0) * eps0
    return x_hat, rng

print("Sampling from trained model…")
rng, sample_rng = jax.random.split(rng)
x_hat, _ = sample_fn(state.params, sample_rng, 10_000)

# -----------------------------------------------------------------------------
# 5) Plot
# -----------------------------------------------------------------------------
plot_MoG40(mog40.log_prob, np.array(x_hat),
           title="identity sampled x_hat on MoG40 contours")

plt.figure(figsize=(6,4))
plt.plot(np.arange(model.T),
         np.array(gamma_t_jax[:,0]**2),
         linewidth=1)
plt.xlabel("t")
plt.ylabel(r"$\gamma_t^2$")
plt.title("identity reverse variances $\gamma_t^2$ over t")
plt.tight_layout()
plt.show()
