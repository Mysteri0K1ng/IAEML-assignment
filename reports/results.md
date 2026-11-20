# Measured experiment results

Author: Nikita Shankin. Recorded October 30, 2026.

## Held-out comparison

Each policy uses the same 128 reset seeds, 1000-1127, on map 3. Learned policies use deterministic tanh(mean) actions. The initial network is the exact network saved before seed-0 training.

| Policy | Successes | Collisions | Timeouts | Success rate | Mean return | Mean actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Heuristic | 90 | 38 | 0 | 70.31% | 38.94 | 62.31 |
| Initial network | 0 | 0 | 128 | 0.00% | -2.39 | 240.00 |
| Learned | 117 | 11 | 0 | 91.41% | 50.26 | 46.19 |
| Random | 1 | 99 | 28 | 0.78% | 0.26 | 160.56 |
| Zero | 0 | 0 | 128 | 0.00% | -2.40 | 240.00 |

The waypoint heuristic does not react to moving hazards. Its weaker result does not establish superiority over a tuned planner or controller. Zero actions avoid collisions here because the hazard does not cross the stationary start region, but they never complete the task.

## Repeated training seeds

Each run uses 300 updates and 1,228,800 transitions with identical hyperparameters. Seed 0 remains the primary checkpoint and video; it was chosen by seed number, not held-out performance.

| Training seed | Successes / 128 | Success rate | Mean return | Training seconds |
| --- | ---: | ---: | ---: | ---: |
| 0 | 117 | 91.41% | 50.26 | 31.40 |
| 1 | 119 | 92.97% | 51.03 | 30.97 |
| 2 | 113 | 88.28% | 48.54 | 29.65 |

Mean success rate across the 3 training seeds: **90.89%**, sample standard deviation **2.39 percentage points**. Three seeds provide only a small estimate of training variability. The reset seeds are shared across runs, so the 384 outcomes should not be treated as fully independent evidence of generalization.

Training seconds cover the update loop, including its first JIT compilation, host logging, and periodic checkpoint writes. They exclude dependency installation, initial route/network setup, evaluation, plotting, and video recording. Runs executed on Windows 11, Python 3.13.4, JAX 0.12.0.1, CPU device `cpu:0`; see each run's `run.json` for full metadata.

## Environment benchmark

For 32 environments and 128 actions each (4096 transitions), the first compiled call took 0.3937 s. Across 5 warmed repeats, the compiled scan averaged 7.63 ms and the Python-dispatched vectorized loop averaged 27.43 ms: an observed **3.59x** ratio on this machine.

Both paths retain rewards and all observations and synchronize outputs. Their numerical outputs are checked for agreement. The Python path includes stacking the per-step outputs. The measurement includes absorbing episodes and is an environment microbenchmark; it is not full PPO throughput. The five repeat timings are retained in [benchmark.json](../results/benchmark.json).

## Videos and interpretation

- [environment.mp4](../results/environment.mp4): heuristic policy, reset seed 0, 70 actions, return 54.32, success=true, collision=false.
- [learned.mp4](../results/learned.mp4): learned policy, reset seed 0, 46 actions, return 53.34, success=true, collision=false.

The environment video uses the fixed template start; the learned-policy video uses randomized reset seed 0. They are reproducible demonstrations, not samples of aggregate reliability. The learned taxi enters the goal at its speed limit; it has not learned a parking or stopping objective.

## Scope and validation

The final test suite passes 15 tests, including physical transitions, collision geometry, episode semantics, JIT/vectorization, the three scaled original-map static routes, GAE bootstrapping, checkpoint restoration, and a short PPO weight-update check. The static-route tests do not certify collision-free execution of the bicycle controller on those original maps.

These experiments do not measure unseen-map generalization, higher-speed tyre dynamics, or a recurrent policy. The remaining 7-12% collision rate across the three trained policies is a substantial limitation.
