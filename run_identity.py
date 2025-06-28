import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from tqdm import trange
import matplotlib.pyplot as plt

from avae import Identity
from datasets.mog40 import GMM, plot_MoG40

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

# 1.3) Optax Adam optimizer with better learning rate schedule
schedule = optax.cosine_decay_schedule(init_value=2e-4, decay_steps=100000, alpha=0.1)
tx = optax.chain(
    optax.clip_by_global_norm(1.0),  # gradient clipping
    optax.adam(schedule, b1=0.9, b2=0.999, eps=1e-8)
)
state = train_state.TrainState.create(apply_fn=model.apply,
                                      params=params,
                                      tx=tx)

# 1.4) Precompute initial w, q, and gamma_t - keep everything on device
scheduler_attrs = model.apply(variables, method=model.get_scheduler_attrs)
c0 = scheduler_attrs['c0']
sigma_cond = scheduler_attrs['sigma_cond']

# Initialize weights and probabilities on device
w_init = jnp.log1p(c0**2 * model.z_dim / sigma_cond)
q_init = w_init / jnp.sum(w_init)
gamma_t_init = jnp.broadcast_to(scheduler_attrs['sigmas'][:, None], (model.T, model.z_dim))

# Training hyperparameters
batch_size = 128
num_iters = 1_000
S = 12
print_every = 100
update_schedule_every = 10  # Update schedules less frequently

# Compile the batch sampling function
@jax.jit
def sample_batch(key):
    """Pre-compiled batch sampling from MoG"""
    # Use JAX random instead of numpy for better performance
    mix_idx = jax.random.categorical(key, mog40.log_weights, shape=(batch_size,))
    key, subkey = jax.random.split(key)
    eps = jax.random.normal(subkey, (batch_size, 2))

    # Vectorized sampling
    locs = mog40.locs[mix_idx]  # [batch_size, 2]
    scales = jnp.diagonal(mog40.scale_tril[mix_idx], axis1=1, axis2=2)  # [batch_size, 2]
    samples = locs + scales * eps
    return samples

@jax.jit
def train_step(state, x0, q, gamma_t, rng):
    """Optimized training step with better memory usage"""
    rng, rng_t, rng_eps, rng_f = jax.random.split(rng, 4)

    # Sample timesteps - use more efficient sampling
    t = jax.random.categorical(rng_t, jnp.log(q), shape=(x0.shape[0],))
    tp1 = jnp.minimum(t + 1, model.T - 1)

    # Forward noising
    eps_tp1 = jax.random.normal(rng_eps, shape=x0.shape)

    def loss_fn(params):
        # q(z_{t+1}|x0)
        z_tp1 = state.apply_fn({'params': params},
                               x0, tp1, eps_tp1,
                               method=Identity.q)

        # energy f_t with optimized antithetic sampling
        f, A, B_hat = state.apply_fn({'params': params},
                                     z_tp1, x0, t, rng_f,
                                     antithetic_sampling=True, S=S,
                                     method=Identity.f_theta_t)

        # Compute loss more efficiently
        log_e = jnp.log1p(B_hat / A)
        per_sample_loss = jnp.sum(log_e, axis=1) / q[t]
        loss = 0.5 * jnp.mean(per_sample_loss)

        # Return additional info for schedule updates
        return loss, (f, log_e, t, per_sample_loss)

    # Compute gradients
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (f, log_e, t_out, per_sample_loss)), grads = grad_fn(state.params)

    # Update parameters
    state = state.apply_gradients(grads=grads)

    return state, loss, f, log_e, t_out, per_sample_loss, rng

@jax.jit
def update_schedules(gamma_t, w, f_vals, log_e_vals, t_vals, per_sample_loss):
    """Vectorized schedule updates on device using scatter operations"""

    # Use scatter_add for efficient updates instead of boolean indexing
    def update_gamma_for_timestep(gamma_t_current, t_idx):
        mask = (t_vals == t_idx).astype(jnp.float32)
        count = jnp.sum(mask)

        # Compute weighted average using scatter operations
        def has_samples():
            # Weighted sum of f_vals for this timestep
            weights = mask[:, None] / jnp.maximum(count, 1.0)  # Avoid division by zero
            f_mean = jnp.sum(f_vals * weights, axis=0)
            return gamma_t_current.at[t_idx].set(f_mean)

        def no_samples():
            return gamma_t_current

        return jax.lax.cond(count > 0, has_samples, no_samples)

    def update_w_for_timestep(w_current, t_idx):
        mask = (t_vals == t_idx).astype(jnp.float32)
        count = jnp.sum(mask)

        def has_samples():
            # Weighted average for this timestep - sum over both batch and feature dims
            weights = mask / jnp.maximum(count, 1.0)  # [batch_size]
            log_e_weighted = log_e_vals * weights[:, None]  # [batch_size, z_dim]
            log_e_mean = jnp.sum(log_e_weighted)  # Sum over all dimensions
            return w_current.at[t_idx].set(log_e_mean)

        def no_samples():
            return w_current

        return jax.lax.cond(count > 0, has_samples, no_samples)

    # Update gamma_t and w using fori_loop for efficiency
    gamma_t_new = gamma_t
    w_new = w

    # Use fori_loop for better performance with static loop bounds
    def update_gamma_step(i, gamma_t_carry):
        return update_gamma_for_timestep(gamma_t_carry, i)

    def update_w_step(i, w_carry):
        return update_w_for_timestep(w_carry, i)

    gamma_t_new = jax.lax.fori_loop(0, model.T, update_gamma_step, gamma_t_new)
    w_new = jax.lax.fori_loop(0, model.T, update_w_step, w_new)

    # Renormalize probabilities
    q_new = w_new / jnp.sum(w_new)

    return gamma_t_new, w_new, q_new

# -----------------------------------------------------------------------------
# 3) Optimized training loop
# -----------------------------------------------------------------------------
print("Starting training...")

# Initialize schedules
gamma_t_current = gamma_t_init
w_current = w_init
q_current = q_init

# Pre-compile everything
print("Compiling functions...")
rng, sample_key = jax.random.split(rng)
dummy_batch = sample_batch(sample_key)
rng, step_key = jax.random.split(rng)

# Warmup compilation
_, _, _, _, _, _, _ = train_step(state, dummy_batch, q_current, gamma_t_current, step_key)
print("Compilation complete. Starting training...")
# Initialize running loss before the loop
running_loss = 0.0
pbar = trange(num_iters)
for i in pbar:
    # Sample batch efficiently
    rng, sample_key = jax.random.split(rng)
    x0_batch = sample_batch(sample_key)

    # Training step
    rng, step_key = jax.random.split(rng)
    state, loss, f_val, log_e_val, t_idx, per_sample_loss, rng = train_step(
        state, x0_batch, q_current, gamma_t_current, step_key)

    # Update schedules less frequently for better performance
    if (i + 1) % update_schedule_every == 0:
        gamma_t_current, w_current, q_current = update_schedules(
            gamma_t_current, w_current, f_val, log_e_val, t_idx, per_sample_loss)

    running_loss += loss.item()
    if (i + 1) % print_every == 0:
        avg_loss = running_loss / print_every
        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
        running_loss = 0.0

# -----------------------------------------------------------------------------
# 4) Optimized sampling
# -----------------------------------------------------------------------------
print("Sampling from trained model...")
rng, sample_rng = jax.random.split(rng)

# Use the model's built-in sampling method instead of a separate function
x_hat = model.apply({'params': state.params}, 10_000, sample_rng, method=model.sample)

# -----------------------------------------------------------------------------
# 5) Plotting
# -----------------------------------------------------------------------------
plot_MoG40(mog40.log_prob, np.array(x_hat),
           title="Identity sampled x_hat on MoG40 contours")

plt.figure(figsize=(6,4))
plt.plot(np.arange(model.T),
         np.array(gamma_t_current[:,0]**2),
         linewidth=1)
plt.xlabel("t")
plt.ylabel(r"$\gamma_t^2$")
plt.title(r"Identity reverse variances $\gamma_t^2$ over t")
plt.tight_layout()
plt.show()
