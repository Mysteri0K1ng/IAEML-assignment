"""Separate compilation from synchronized steady-state environment timing."""
import argparse
import json
from pathlib import Path
import time
import jax
import jax.numpy as jnp
from env.robotaxi import RoboTaxiEnv


def benchmark(num_envs=32, steps=128, repeats=5):
    p, template = RoboTaxiEnv.init_params(jax.random.PRNGKey(0))
    keys = jax.random.split(jax.random.PRNGKey(1), num_envs)
    _, states = jax.vmap(lambda k: RoboTaxiEnv.reset(k, p, template))(keys)
    action = jnp.tile(jnp.array([[.2, .1]]), (num_envs, 1))
    vector_step = jax.jit(jax.vmap(lambda k, s, a: RoboTaxiEnv.step(k, s, a, p)))

    @jax.jit
    def compiled(states):
        def body(states, _):
            observation, states, reward, _, _ = vector_step(keys, states, action)
            return states, (reward, observation.as_vector())
        return jax.lax.scan(body, states, None, length=steps)

    started = time.perf_counter()
    jax.block_until_ready(compiled(states))
    first = time.perf_counter() - started
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(compiled(states))
        timings.append(time.perf_counter() - started)
    # Same physics and vectorization, dispatched from a Python timestep loop.
    # Warm the whole path, including stacking, to exclude its compilations too.
    def dispatched(states):
        current = states
        rewards, observations = [], []
        for _ in range(steps):
            observation, current, reward, _, _ = vector_step(keys, current, action)
            rewards.append(reward)
            observations.append(observation.as_vector())
        return current, (jnp.stack(rewards), jnp.stack(observations))

    expected = jax.block_until_ready(dispatched(states))
    actual = jax.block_until_ready(compiled(states))
    # Performance comparisons are only meaningful for equivalent outputs.
    import numpy as np
    for lhs, rhs in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        np.testing.assert_allclose(lhs, rhs, rtol=1e-5, atol=1e-5)
    python_timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(dispatched(states))
        python_timings.append(time.perf_counter() - started)
    python_loop = sum(python_timings) / len(python_timings)
    steady = sum(timings) / len(timings)
    return {"num_envs": num_envs, "steps": steps, "repeats": repeats,
            "first_call_seconds": first, "compiled_seconds": timings,
            "mean_compiled_seconds": steady, "python_loop_seconds": python_loop,
            "python_loop_repeats_seconds": python_timings,
            "compiled_steps_per_second": num_envs * steps / steady,
            "python_to_compiled_ratio": python_loop / steady,
            "device": str(jax.devices()[0]),
            "note": "Environment only; both paths retain rewards and all observations, include absorbing states and dispatch, and exclude resets and PPO. Python loop includes output stacking."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/benchmark.json")
    args = parser.parse_args()
    result = benchmark()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
