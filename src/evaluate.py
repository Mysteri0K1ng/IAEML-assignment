"""Evaluate frozen policies on reproducible, held-out reset seeds."""
import argparse
from dataclasses import replace
from functools import partial
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from algorithms.ppo import load_checkpoint, network
from env.robotaxi import RoboTaxiEnv, RoboTaxiParams


def heuristic(state, params):
    """Simple waypoint pure pursuit; reference controller, never used by PPO."""
    distances = jnp.linalg.norm(state.path_array - state.agent_pos, axis=-1)
    idx = jnp.argmin(distances)
    target = state.path_array[jnp.minimum(idx + 3, params.path_length - 1)]
    delta = target - state.agent_pos
    forward = state.agent_forward_dir
    local = jnp.array([jnp.dot(delta, forward), -delta[0] * forward[1] + delta[1] * forward[0]])
    angle = jnp.arctan2(local[1], local[0])
    steering = jnp.arctan2(2. * params.wheelbase * jnp.sin(angle), jnp.maximum(jnp.linalg.norm(delta), 1.))
    target_speed = jnp.minimum(2.5, jnp.maximum(.4, jnp.linalg.norm(state.goal_pos - state.agent_pos)))
    throttle = jnp.clip(1.5 * (target_speed - state.speed), -1., 1.)
    return jnp.array([throttle, jnp.clip(steering / params.max_steer, -1., 1.)])


def make_environment(checkpoint=None, map_id=None, randomized=True):
    weights = None
    if checkpoint:
        weights, metadata = load_checkpoint(checkpoint)
        saved_id = metadata["config"]["map_id"]
        chosen_id = saved_id if map_id is None else map_id
        params, template = RoboTaxiEnv.init_params(jax.random.PRNGKey(0), map_id=chosen_id)
        if chosen_id == saved_id:
            environment = metadata["environment"].copy()
            environment["map_height_width"] = tuple(environment["map_height_width"])
            params = RoboTaxiParams(**environment)
    else:
        params, template = RoboTaxiEnv.init_params(jax.random.PRNGKey(0), map_id=3 if map_id is None else map_id)
    params = replace(params, randomize_reset=params.randomize_reset and randomized)
    return weights, params, template


def evaluate(checkpoint=None, episodes=128, seed=1000, policy="learned", map_id=None):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if policy == "learned" and checkpoint is None:
        raise ValueError("Learned policy needs --checkpoint")
    weights, params, template = make_environment(checkpoint, map_id)
    keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(seed, seed + episodes))
    obs, states = jax.vmap(lambda k: RoboTaxiEnv.reset(k, params, template))(keys)

    @jax.jit
    def simulate(states, obs):
        def body(carry, t):
            states, obs, returns, lengths, ended = carry
            if policy == "learned":
                actions = jnp.tanh(network(weights, obs.as_vector())[0])
            elif policy == "random":
                step_keys = jax.vmap(lambda k: jax.random.fold_in(k, t))(keys)
                actions = jax.vmap(lambda k: jax.random.uniform(k, (2,), minval=-1., maxval=1.))(step_keys)
            elif policy == "heuristic":
                actions = jax.vmap(lambda s: heuristic(s, params))(states)
            else:
                actions = jnp.zeros((episodes, 2))
            obs, states, reward, done, info = jax.vmap(lambda k, s, a: RoboTaxiEnv.step(k, s, a, params))(keys, states, actions)
            returns += jnp.where(ended, 0., reward)
            lengths += ~ended
            return (states, obs, returns, lengths, ended | done), None
        result, _ = jax.lax.scan(body, (states, obs, jnp.zeros(episodes), jnp.zeros(episodes, jnp.int32), jnp.zeros(episodes, bool)), jnp.arange(params.max_steps_in_episode))
        return result

    states, _, returns, lengths, _ = jax.device_get(simulate(states, obs))
    actual_map_id = map_id
    if actual_map_id is None:
        actual_map_id = json.loads(Path(checkpoint).with_suffix(".json").read_text())["config"]["map_id"] if checkpoint else 3
    result = {"policy": policy, "checkpoint": str(checkpoint) if checkpoint else None,
              "episodes": episodes, "seed_start": seed, "seed_end": seed + episodes - 1,
              "map_id": actual_map_id,
              "successes": int(states.success.sum()), "collisions": int(states.collision.sum()),
              "timeouts": int(states.truncated.sum()), "success_rate": float(states.success.mean()),
              "collision_rate": float(states.collision.mean()), "mean_return": float(returns.mean()),
              "std_return": float(returns.std()), "mean_length": float(lengths.mean()),
              "mean_final_distance": float(np.linalg.norm(states.agent_pos - states.goal_pos, axis=-1).mean())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--map-id", type=int)
    parser.add_argument("--policy", choices=["learned", "random", "zero", "heuristic"], default="learned")
    parser.add_argument("--output", default="results/evaluation.json")
    args = vars(parser.parse_args())
    path = Path(args.pop("output"))
    result = evaluate(**args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
