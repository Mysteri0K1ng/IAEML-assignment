import jax
import jax.numpy as jnp
import numpy as np
from algorithms.ppo import PPOConfig, advantages, gaussian_log_prob, init_network, load_checkpoint, network, save_checkpoint, train
from env.robotaxi import RoboTaxiParams


def test_gae_bootstraps_truncation_without_crossing_reset():
    rewards = jnp.array([[1.], [2.], [100.]])
    values = jnp.zeros_like(rewards)
    next_values = jnp.array([[3.], [4.], [5.]])
    terminated = jnp.array([[False], [False], [True]])
    done = jnp.array([[False], [True], [True]])
    adv, returns = advantages(rewards, values, next_values, terminated, done, .9, 1.)
    np.testing.assert_allclose(adv[:, 0], [1 + .9 * 3 + .9 * (2 + .9 * 4), 2 + .9 * 4, 100], rtol=1e-6)


def test_checkpoint_roundtrip(tmp_path):
    weights = init_network(jax.random.PRNGKey(0), 24, 16)
    target = tmp_path / "model.npz"
    save_checkpoint(target, weights, PPOConfig(hidden_size=16), RoboTaxiParams())
    loaded, metadata = load_checkpoint(target)
    a, b = network(weights, jnp.zeros(24)), network(loaded, jnp.zeros(24))
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)
    assert metadata["format_version"] == 1


def test_training_smoke_updates_actor(tmp_path):
    config = PPOConfig(updates=2, num_envs=4, rollout_steps=8, epochs=2, minibatch_size=16, hidden_size=16)
    trained, history = train(config, tmp_path)
    initial, _ = load_checkpoint(tmp_path / "initial.npz")
    assert len(history) == 2
    assert all(np.isfinite(row["value_loss"]) for row in history)
    assert not np.array_equal(initial["actor_w2"], trained["actor_w2"])
