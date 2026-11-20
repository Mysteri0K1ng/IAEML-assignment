# IAEML: Robo-taxi environment and PPO

Author: **Nikita Shankin**. Based on the [InnoRL starter repository](https://github.com/InnoRL/IAEML-assignment).

This project implements a JAX bicycle-model vehicle environment and a JAX/Optax PPO experiment. It includes tests, training logs, baseline comparisons, a numeric inference checkpoint, headless MP4 recordings, and English technical reports.

The reports include Nikita's personal reflections, supplied on November 20, 2026.

## Deliverables

| Requirement | Location |
| --- | --- |
| JAX environment inheriting from BaseEnv | `src/env/robotaxi.py` |
| Acceleration, steering, drag, sensing, reward | `src/env/robotaxi.py` |
| Environment report | [Assignment 1](reports/assignment1.md) |
| Environment video | [Reference-controller MP4](results/environment.mp4) |
| PPO implementation | `src/algorithms/ppo.py` |
| Experimental setup and optimizations | [Assignment 2](reports/assignment2.md) |
| Measured results | [Results](reports/results.md) |
| Final model and metadata | `results/ppo_seed0/policy.npz`, `policy.json` |
| Learned-policy video | [PPO MP4](results/learned.mp4) |
| Training and evaluation figures | `results/ppo_seed0/training.png`, `evaluation.png` |

## Setup

Python 3.13 on Windows CPU was used for the recorded experiment. Exact tested packages are pinned in `requirements.txt`; `requirements-lock.txt` also records transitive dependencies. Other Python/platform combinations may need compatible wheels. JAX CPU works without a GPU. Installation instructions for other hardware are in the [official JAX documentation](https://docs.jax.dev/en/latest/installation.html).

PowerShell, from the repository directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = (Join-Path $PWD 'src')
.\.venv\Scripts\python.exe -m pytest -q
```

On Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src"
python -m pytest -q
```

The project also supports `python -m pip install -e .`. Then `PYTHONPATH` is unnecessary. Commands below use `python` from the activated environment. In PowerShell either activate it with `.\.venv\Scripts\Activate.ps1` or replace `python` with `.\.venv\Scripts\python.exe`.

## Run and reproduce

Manual controls use a separate frontend with correctly sized circle graphics:

```bash
python -m demo --interactive
```

W/S or up/down: accelerate/brake. A/D or left/right: steer. R: reset. Esc: exit. The default is the new training map 3; `--map-id 0`, `1`, or `2` selects a scaled original map. Original unit-spaced maps can be accessed through `RoboTaxiEnv.init_params(world_scale=1)`.

Record the environment reference controller and train PPO:

```bash
python -m demo --policy heuristic --seed 0 --fixed-start --output results/environment.mp4
python -m algorithms.ppo --seed 0 --updates 300 --output results/ppo_seed0
```

For a quick pipeline check, use `--updates 2 --num-envs 4 --rollout-steps 8 --epochs 2 --minibatch-size 16 --output results/smoke`. This short run is not expected to learn a useful policy.

Evaluate saved and baseline policies on identical reset seeds:

```bash
python -m evaluate --checkpoint results/ppo_seed0/policy.npz --episodes 128 --seed 1000 --output results/ppo_seed0/eval_learned.json
python -m evaluate --checkpoint results/ppo_seed0/initial.npz --episodes 128 --seed 1000 --output results/ppo_seed0/eval_initial.json
python -m evaluate --policy random --episodes 128 --seed 1000 --output results/ppo_seed0/eval_random.json
python -m evaluate --policy heuristic --episodes 128 --seed 1000 --output results/ppo_seed0/eval_heuristic.json
python -m evaluate --policy zero --episodes 128 --seed 1000 --output results/ppo_seed0/eval_zero.json
python -m demo --checkpoint results/ppo_seed0/policy.npz --policy learned --seed 0 --output results/learned.mp4
python -m benchmark
python scripts/summarize_results.py
```

Recorded results use the final checkpoints from three training seeds (0, 1, 2). To reproduce the repeats, train with `--seed 1 --output results/ppo_seed1` and `--seed 2 --output results/ppo_seed2`, then evaluate each checkpoint on the same 128 reset seeds. Run `scripts/summarize_results.py` after all evaluations to regenerate the results appendix. Seed 0 is the primary checkpoint and video.

The results measure randomized resets on one fixed map and goal. They do not establish generalization to the original maps or moving-obstacle speeds outside the experiment. The inference snapshot does not include state for exact training resumption. Retraining may differ slightly across JAX versions or hardware.

## Structure

```text
src/env/base.py          original starter, unchanged
src/env/tasks.py         original maps, unchanged
src/env/robotaxi.py      new vehicle state, sensing, reward and planner
src/algorithms/ppo.py    PPO rollout, GAE, loss, optimizer and checkpoint
src/evaluate.py         held-out batched evaluation and reference controller
src/demo.py             manual control and headless MP4/GIF recording
src/benchmark.py        synchronized environment timing
tests/                  physics, episode semantics, GAE and training checks
reports/                English reports and measured results
results/                logs, figures, checkpoint, videos and provenance
```

The upstream `src/main.py` remains a demonstration of `BaseEnv` and its original action semantics. Use `python -m demo` for this vehicle implementation. All new simulation and optimization logic uses JAX; Pillow and imageio are used only for rendering, and NumPy/JSON for serialization and analysis.

## Attribution

The original base class, coordinate helpers, ray-intersection geometry, maps, and renderer are supplied by [InnoRL/IAEML-assignment](https://github.com/InnoRL/IAEML-assignment). The new planner, bicycle environment, PPO trainer, evaluation, tests, and recorder extend that scaffold. Algorithm and documentation references appear in the reports. `results/source.json` identifies the upstream revision and artifact checksums.
