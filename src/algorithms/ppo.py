"""Small, self-contained PPO: JAX arrays, Optax, no third-party trainer.

The Gaussian is sampled in latent space and transformed with tanh. PPO ratios
can use latent log probabilities because the identical transform Jacobians
cancel. Entropy regularization uses the unsquashed Gaussian approximation.
"""
import argparse
from dataclasses import asdict, dataclass
from functools import partial
import json
from pathlib import Path
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from env.robotaxi import RoboTaxiEnv


@dataclass(frozen=True)
class PPOConfig:
    seed: int = 0
    updates: int = 300
    num_envs: int = 32
    rollout_steps: int = 128
    epochs: int = 4
    minibatch_size: int = 512
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.001
    max_grad_norm: float = 0.5
    hidden_size: int = 64
    map_id: int = 3


def init_network(key, obs_dim, hidden=64):
    weights = {}
    for prefix, out_dim in (("actor", 2), ("critic", 1)):
        for layer, (din, dout) in enumerate(((obs_dim, hidden), (hidden, hidden), (hidden, out_dim))):
            key, k = jax.random.split(key)
            scale = (0.01 if prefix == "actor" else 1.) if layer == 2 else np.sqrt(2.)
            weights[f"{prefix}_w{layer}"] = jax.nn.initializers.orthogonal(scale)(k, (din, dout))
            weights[f"{prefix}_b{layer}"] = jnp.zeros(dout)
    weights["log_std"] = jnp.full((2,), -0.5)
    return weights


def network(weights, obs):
    outputs = []
    for prefix in ("actor", "critic"):
        x = obs
        for i in range(3):
            x = x @ weights[f"{prefix}_w{i}"] + weights[f"{prefix}_b{i}"]
            if i < 2:
                x = jnp.tanh(x)
        outputs.append(x)
    return outputs[0], jnp.clip(weights["log_std"], -5., 1.), outputs[1][..., 0]


def gaussian_log_prob(latent, mean, log_std):
    return -0.5 * jnp.sum(((latent - mean) / jnp.exp(log_std))**2 + 2. * log_std + np.log(2. * np.pi), axis=-1)


def advantages(rewards, values, next_values, terminated, done, gamma, lam):
    """Bootstrap time limits, but never carry GAE into a reset episode."""
    delta = rewards + gamma * (~terminated) * next_values - values

    def scan(gae, inputs):
        d, ended = inputs
        gae = d + gamma * lam * (~ended) * gae
        return gae, gae

    _, adv = jax.lax.scan(scan, jnp.zeros_like(values[0]), (delta, done), reverse=True)
    return adv, adv + values


def save_checkpoint(path, weights, config, env_params):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **jax.device_get(weights))
    path.with_suffix(".json").write_text(json.dumps({"format_version": 1, "config": asdict(config),
                                                   "environment": asdict(env_params)}, indent=2) + "\n")


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        weights = {k: jnp.asarray(data[k]) for k in data.files}
    metadata = json.loads(Path(path).with_suffix(".json").read_text())
    return weights, metadata


def make_update(config, env_params, template, optimizer):
    c, p = config, env_params
    batch_size = c.num_envs * c.rollout_steps
    if batch_size % c.minibatch_size:
        raise ValueError("Rollout batch must be divisible by minibatch size")
    reset_many = jax.vmap(lambda key: RoboTaxiEnv.reset(key, p, template))
    step_many = jax.vmap(lambda key, state, action: RoboTaxiEnv.step(key, state, action, p))

    @jax.jit
    def update(carry):
        weights, opt_state, states, obs, key, episode_returns = carry

        def rollout(carry, _):
            states, obs, key, episode_returns = carry
            key, ka, ks, kr = jax.random.split(key, 4)
            mean, log_std, value = network(weights, obs)
            latent = mean + jnp.exp(log_std) * jax.random.normal(ka, mean.shape)
            log_prob = gaussian_log_prob(latent, mean, log_std)
            next_obs, next_states, reward, done, info = step_many(jax.random.split(ks, c.num_envs), states, jnp.tanh(latent))
            next_vec = next_obs.as_vector()
            next_value = network(weights, next_vec)[2]
            fresh_obs, fresh_states = reset_many(jax.random.split(kr, c.num_envs))
            states = jax.tree.map(lambda old, fresh: jnp.where(done.reshape((c.num_envs,) + (1,) * (old.ndim - 1)), fresh, old), next_states, fresh_states)
            reset_vec = fresh_obs.as_vector()
            new_obs = jnp.where(done[:, None], reset_vec, next_vec)
            total_returns = episode_returns + reward
            stats = jnp.stack([jnp.sum(done), jnp.sum(info["success"]), jnp.sum(info["collision"]),
                               jnp.sum(info["truncated"]), jnp.sum(jnp.where(done, total_returns, 0.))])
            transition = (obs, latent, log_prob, value, reward, next_value, info["terminated"], done)
            return (states, new_obs, key, jnp.where(done, 0., total_returns)), (transition, stats)

        (states, obs, key, episode_returns), (trajectory, stats) = jax.lax.scan(rollout, (states, obs, key, episode_returns), None, length=c.rollout_steps)
        observations, latent, old_log_prob, values, reward, next_values, terminated, done = trajectory
        adv, returns = advantages(reward, values, next_values, terminated, done, c.gamma, c.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        data = (observations, latent, old_log_prob, values, adv, returns)
        data = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), data)

        def loss(weights, batch):
            o, a, old_lp, old_v, advantage, targets = batch
            mean, log_std, value = network(weights, o)
            lp = gaussian_log_prob(a, mean, log_std)
            log_ratio = lp - old_lp
            ratio = jnp.exp(log_ratio)
            surrogate = jnp.minimum(ratio * advantage, jnp.clip(ratio, 1. - c.clip_epsilon, 1. + c.clip_epsilon) * advantage)
            policy_loss = -surrogate.mean()
            clipped_value = old_v + jnp.clip(value - old_v, -c.clip_epsilon, c.clip_epsilon)
            value_loss = 0.5 * jnp.maximum((value - targets)**2, (clipped_value - targets)**2).mean()
            entropy = jnp.sum(log_std + 0.5 * np.log(2. * np.pi * np.e))
            total = policy_loss + c.value_coefficient * value_loss - c.entropy_coefficient * entropy
            kl = jnp.mean((ratio - 1.) - log_ratio)
            clip_fraction = jnp.mean(jnp.abs(ratio - 1.) > c.clip_epsilon)
            return total, jnp.array([policy_loss, value_loss, entropy, kl, clip_fraction])

        def epoch(carry, _):
            weights, opt_state, key = carry
            key, perm_key = jax.random.split(key)
            permutation = jax.random.permutation(perm_key, batch_size)
            batches = jax.tree.map(lambda x: x[permutation].reshape((-1, c.minibatch_size) + x.shape[1:]), data)

            def minibatch(carry, batch):
                weights, opt_state = carry
                (_, metrics), grads = jax.value_and_grad(loss, has_aux=True)(weights, batch)
                updates, opt_state = optimizer.update(grads, opt_state, weights)
                weights = optax.apply_updates(weights, updates)
                return (weights, opt_state), metrics

            (weights, opt_state), metrics = jax.lax.scan(minibatch, (weights, opt_state), batches)
            return (weights, opt_state, key), metrics.mean(axis=0)

        (weights, opt_state, key), losses = jax.lax.scan(epoch, (weights, opt_state, key), None, length=c.epochs)
        return (weights, opt_state, states, obs, key, episode_returns), (stats.sum(axis=0), losses.mean(axis=0))
    return update


def train(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    c = config
    if min(c.updates, c.num_envs, c.rollout_steps, c.epochs, c.minibatch_size) < 1:
        raise ValueError("Training dimensions must be positive")
    p, template = RoboTaxiEnv.init_params(jax.random.PRNGKey(c.seed), map_id=c.map_id)
    key, kn, kr = jax.random.split(jax.random.PRNGKey(c.seed), 3)
    observations, states = jax.vmap(lambda k: RoboTaxiEnv.reset(k, p, template))(jax.random.split(kr, c.num_envs))
    obs = observations.as_vector()
    weights = init_network(kn, obs.shape[-1], c.hidden_size)
    save_checkpoint(output / "initial.npz", weights, c, p)
    optimizer = optax.chain(optax.clip_by_global_norm(c.max_grad_norm), optax.adam(c.learning_rate, eps=1e-5))
    update = make_update(c, p, template, optimizer)
    carry = (weights, optimizer.init(weights), states, obs, key, jnp.zeros(c.num_envs))
    history = []
    started = time.perf_counter()
    for i in range(c.updates):
        carry, (stats, losses) = update(carry)
        stats, losses = jax.device_get((stats, losses))
        if not (np.isfinite(stats).all() and np.isfinite(losses).all()):
            raise FloatingPointError(f"Non-finite training metrics at update {i + 1}")
        n, success, collision, timeout, total_return = stats.tolist()
        row = {"update": i + 1, "steps": (i + 1) * c.num_envs * c.rollout_steps,
               "episodes": int(n), "successes": int(success), "collisions": int(collision), "timeouts": int(timeout),
               "mean_completed_return": total_return / max(n, 1), "elapsed_seconds": time.perf_counter() - started}
        row.update(dict(zip(("policy_loss", "value_loss", "latent_entropy", "approx_kl", "clip_fraction"), losses.tolist())))
        history.append(row)
        if i == 0 or (i + 1) % 10 == 0 or i + 1 == c.updates:
            print(f"update={i+1:4d} steps={row['steps']:8d} episodes={int(n):3d} success={success/max(n,1):.2f} return={row['mean_completed_return']:.2f} elapsed={row['elapsed_seconds']:.1f}s", flush=True)
            save_checkpoint(output / "policy.npz", carry[0], c, p)
            (output / "training.json").write_text(json.dumps(history, indent=2) + "\n")
    environment = {"python": platform.python_version(), "platform": platform.platform(), "jax": jax.__version__,
                   "devices": [str(d) for d in jax.devices()], "config": asdict(c), "seconds": time.perf_counter() - started}
    (output / "run.json").write_text(json.dumps(environment, indent=2) + "\n")
    return carry[0], history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/ppo_seed0")
    for name, field in PPOConfig.__dataclass_fields__.items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(field.default), default=field.default)
    args = vars(parser.parse_args())
    output = args.pop("output")
    train(PPOConfig(**args), output)


if __name__ == "__main__":
    main()
