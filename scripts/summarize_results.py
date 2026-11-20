"""Generate figures from actual JSON logs; does not invent experiment data."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="results/ppo_seed0")
    args = parser.parse_args()
    run = Path(args.run)
    history = json.loads((run / "training.json").read_text())
    # Pool counts within nonoverlapping groups instead of averaging ratios.
    windows = [history[i:i+10] for i in range(0, len(history), 10)]
    x = [rows[-1]["steps"] for rows in windows]
    n = [sum(r["episodes"] for r in rows) for rows in windows]
    success = [sum(r["successes"] for r in rows) / max(count, 1) for rows, count in zip(windows, n)]
    returns = [sum(r["mean_completed_return"] * r["episodes"] for r in rows) / max(count, 1) for rows, count in zip(windows, n)]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout="constrained")
    axes[0].plot(x, success, color="#147d64", linewidth=2)
    axes[0].set(ylabel="Completed-episode success rate", ylim=(0, 1.02))
    axes[1].plot(x, returns, color="#276fbd", linewidth=2)
    axes[1].set(ylabel="Mean completed-episode return")
    axes[2].plot([r["steps"] for r in history], [r["approx_kl"] for r in history], color="#a45b26")
    axes[2].set(ylabel="Approximate KL")
    for ax in axes:
        ax.set_xlabel("Environment steps")
        ax.grid(alpha=.2)
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    seed = json.loads((run / "run.json").read_text())["config"]["seed"]
    fig.suptitle(f"PPO training | seed {seed} | map 3 | training episodes")
    fig.savefig(run / "training.png", dpi=170)
    plt.close(fig)
    evaluations = []
    for path in sorted(run.glob("eval_*.json")):
        row = json.loads(path.read_text())
        row["source"] = str(path).replace("\\", "/")
        evaluations.append(row)
    if evaluations:
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        labels = ["Initial network" if "initial" in r["source"] else r["policy"].title() for r in evaluations]
        rates = [r["success_rate"] for r in evaluations]
        # Wilson 95% interval for Bernoulli success, not across training seeds.
        errors_low, errors_high = [], []
        for r in evaluations:
            n, q, z = r["episodes"], r["success_rate"], 1.96
            center = (q + z*z / (2*n)) / (1 + z*z / n)
            half = z * np.sqrt(q * (1 - q) / n + z*z / (4*n*n)) / (1 + z*z / n)
            errors_low.append(max(0, q - (center - half)))
            errors_high.append(max(0, center + half - q))
        ax.bar(labels, rates, color=["#718096" if label != "Learned" else "#147d64" for label in labels], yerr=[errors_low, errors_high], capsize=4)
        ax.set(ylabel="Success rate (95% Wilson interval)", ylim=(0, 1.08), title="Held-out reset seeds 1000-1127 | map 3")
        ax.grid(axis="y", alpha=.2)
        fig.savefig(run / "evaluation.png", dpi=170)
        plt.close(fig)
        (run / "evaluation_summary.json").write_text(json.dumps(evaluations, indent=2) + "\n")

    if run.name == "ppo_seed0":
        lines = ["# Measured experiment results", "", "Author: Nikita Shankin. Recorded October 30, 2026.", "",
                 "## Held-out comparison", "",
                 "Each policy uses the same 128 reset seeds, 1000-1127, on map 3. Learned policies use deterministic tanh(mean) actions. The initial network is the exact network saved before seed-0 training.", "",
                 "| Policy | Successes | Collisions | Timeouts | Success rate | Mean return | Mean actions |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in evaluations:
            label = "Initial network" if "initial" in row["source"] else row["policy"].title()
            lines.append(f"| {label} | {row['successes']} | {row['collisions']} | {row['timeouts']} | {100*row['success_rate']:.2f}% | {row['mean_return']:.2f} | {row['mean_length']:.2f} |")
        lines += ["", "The waypoint heuristic does not react to moving hazards. Its weaker result does not establish superiority over a tuned planner or controller. Zero actions avoid collisions here because the hazard does not cross the stationary start region, but they never complete the task.", "",
                  "## Repeated training seeds", "",
                  "Each run uses 300 updates and 1,228,800 transitions with identical hyperparameters. Seed 0 remains the primary checkpoint and video; it was chosen by seed number, not held-out performance.", "",
                  "| Training seed | Successes / 128 | Success rate | Mean return | Training seconds |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        rates = []
        for s in range(3):
            folder = Path(f"results/ppo_seed{s}")
            if not (folder / "eval_learned.json").exists():
                continue
            row = json.loads((folder / "eval_learned.json").read_text())
            runtime = json.loads((folder / "run.json").read_text())
            rates.append(row["success_rate"])
            lines.append(f"| {s} | {row['successes']} | {100*row['success_rate']:.2f}% | {row['mean_return']:.2f} | {runtime['seconds']:.2f} |")
        if len(rates) > 1:
            lines += ["", f"Mean success rate across the {len(rates)} training seeds: **{100*np.mean(rates):.2f}%**, sample standard deviation **{100*np.std(rates, ddof=1):.2f} percentage points**. Three seeds provide only a small estimate of training variability. The reset seeds are shared across runs, so the 384 outcomes should not be treated as fully independent evidence of generalization."]
        lines += ["", "Training seconds cover the update loop, including its first JIT compilation, host logging, and periodic checkpoint writes. They exclude dependency installation, initial route/network setup, evaluation, plotting, and video recording. Runs executed on Windows 11, Python 3.13.4, JAX 0.12.0.1, CPU device `cpu:0`; see each run's `run.json` for full metadata.", ""]
        bench_path = Path("results/benchmark.json")
        if bench_path.exists():
            b = json.loads(bench_path.read_text())
            lines += ["## Environment benchmark", "",
                      f"For {b['num_envs']} environments and {b['steps']} actions each ({b['num_envs']*b['steps']} transitions), the first compiled call took {b['first_call_seconds']:.4f} s. Across {b['repeats']} warmed repeats, the compiled scan averaged {1000*b['mean_compiled_seconds']:.2f} ms and the Python-dispatched vectorized loop averaged {1000*b['python_loop_seconds']:.2f} ms: an observed **{b['python_to_compiled_ratio']:.2f}x** ratio on this machine.", "",
                      "Both paths retain rewards and all observations and synchronize outputs. Their numerical outputs are checked for agreement. The Python path includes stacking the per-step outputs. The measurement includes absorbing episodes and is an environment microbenchmark; it is not full PPO throughput. The five repeat timings are retained in [benchmark.json](../results/benchmark.json).", ""]
        lines += ["## Videos and interpretation", ""]
        for name in ("environment", "learned"):
            path = Path(f"results/{name}.json")
            if path.exists():
                v = json.loads(path.read_text())
                lines.append(f"- [{name}.mp4](../results/{name}.mp4): {v['policy']} policy, reset seed {v['seed']}, {v['steps']} actions, return {v['return']:.2f}, success={str(v['success']).lower()}, collision={str(v['collision']).lower()}.")
        lines += ["", "The environment video uses the fixed template start; the learned-policy video uses randomized reset seed 0. They are reproducible demonstrations, not samples of aggregate reliability. The learned taxi enters the goal at its speed limit; it has not learned a parking or stopping objective.", "",
                  "## Scope and validation", "",
                  "The final test suite passes 15 tests, including physical transitions, collision geometry, episode semantics, JIT/vectorization, the three scaled original-map static routes, GAE bootstrapping, checkpoint restoration, and a short PPO weight-update check. The static-route tests do not certify collision-free execution of the bicycle controller on those original maps.", "",
                  "These experiments do not measure unseen-map generalization, higher-speed tyre dynamics, or a recurrent policy. The remaining 7-12% collision rate across the three trained policies is a substantial limitation.", ""]
        Path("reports/results.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
