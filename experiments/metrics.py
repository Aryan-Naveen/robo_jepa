# Copyright (c) 2026 RoboJEPA experiment contributors.
# Licensed under the LICENSE file in the root directory of this source tree.

"""Explicit task and efficiency metric definitions shared by runs and summaries."""
import csv
import json
from pathlib import Path
import statistics

import numpy as np

from .runner import write_json


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def module_metrics(calls):
    if not calls:
        return None
    latency = sum(c["latency_s"] for c in calls)
    samples = sum(c["batch_size"] for c in calls)
    tokens = sum(c["output_tokens"] for c in calls)
    measured_flops = [c["flops"] for c in calls]
    flops = sum(measured_flops) if all(f is not None for f in measured_flops) else None
    peaks = [c["peak_allocated_bytes"] for c in calls if c["peak_allocated_bytes"] is not None]
    increments = [c["peak_increment_bytes"] for c in calls if c["peak_increment_bytes"] is not None]
    # Boundary average, explicitly NOT a continuous time-weighted memory estimate.
    memory = [v for c in calls for v in (c["allocated_before_bytes"], c["allocated_after_bytes"]) if v is not None]
    return {
        "forward_calls": len(calls), "sample_forwards": samples, "output_tokens": tokens,
        "tokens_per_second": tokens / latency if latency else None,
        "latency_total_s": latency, "latency_mean_batch_s": latency / len(calls),
        "latency_amortized_sample_s": latency / samples,
        "flops": flops, "macs": flops / 2 if flops is not None else None,
        "average_boundary_allocated_bytes": mean(memory),
        "peak_allocated_bytes": max(peaks) if peaks else None,
        "peak_increment_bytes": max(increments) if increments else None,
    }


def trial_metrics(trial, config):
    paths = list(trial.glob("episode_*.npz"))
    if len(paths) != 1:
        raise ValueError(f"Expected one completed episode in {trial}, found {len(paths)}")
    efficiency = json.loads((trial / "efficiency.json").read_text())
    with np.load(paths[0], allow_pickle=False) as data:
        rewards = np.asarray(data["env_rewards"], dtype=float)
        if not len(rewards) or not np.all(np.isfinite(rewards)):
            raise ValueError(f"Missing/non-finite episode rewards in {trial}")
        if "env_info_success" not in data:
            raise ValueError("Simulator success flags are required; refusing to infer success from reward")
        successes = np.asarray(data["env_info_success"], dtype=bool)
        if len(successes) != len(rewards):
            raise ValueError("Success flags and reward steps are misaligned")
        indices = np.flatnonzero(successes)
        first_success = int(indices[0]) + 1 if len(indices) else None
        decisions = efficiency["decisions"]
        # execution_prefix=action_substeps=1 is fixed by the public runner:
        # one logged environment action corresponds to one planning decision.
        if len(decisions) != len(rewards):
            raise ValueError("Planning decisions and environment steps differ; cannot map steps-to-success")
        distance_keys = [k for k in data.files if k.endswith("distance_to_goal")]
        key = next((k for k in distance_keys if "push" in k), distance_keys[0] if distance_keys else None)
        final_distance = float(data[key][-1]) if key else None
    module_stats = {name: module_metrics([c for c in efficiency["calls"] if c["module"] == name]) for name in ("encoder", "predictor")}
    per_decision = []
    for d in decisions:
        calls = [c for c in efficiency["calls"] if c["decision"] == d["index"] and c["module"] == "predictor"]
        per_decision.append({**d, "predictor": module_metrics(calls)})
    return {
        "seed": int(trial.name.split("_")[-1]), "task": config["task"], "model_size": config["model_size"],
        "success": bool(len(indices)), "planning_steps_to_success": first_success,
        "planning_steps_executed": len(decisions),
        "task_progress": float(rewards.mean() if config["task"] in ("reach", "object_reach") else rewards[-1]),
        "final_progress": float(rewards[-1]), "mean_progress": float(rewards.mean()),
        "progress_by_step": rewards.tolist(), "final_distance_m": final_distance,
        "planning_latency_total_s": sum(d["latency_s"] for d in decisions),
        "modules": module_stats, "decisions": per_decision,
    }


def flatten(row, prefix=""):
    flat = {}
    for key, value in row.items():
        key = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, key + "."))
        elif not isinstance(value, list):
            flat[key] = value
    return flat


def summarize(output):
    output = Path(output)
    config = json.loads((output / "config.json").read_text())
    rows = []
    failed = []
    for seed in range(config["seed"], config["seed"] + config["trials"]):
        trial = output / f"trial_{seed:05d}"
        if (trial / "metrics.json").exists():
            rows.append(json.loads((trial / "metrics.json").read_text()))
        elif (trial / "failure.json").exists():
            failed.append(seed)
    summary = {
        "requested_trials": config["trials"], "completed_trials": len(rows), "failed_seeds": failed,
        "pending_trials": config["trials"] - len(rows) - len(failed),
        "success_rate": mean([int(r["success"]) for r in rows]),
        "successful_trials": sum(r["success"] for r in rows),
        "mean_task_progress": mean([r["task_progress"] for r in rows]),
        "mean_planning_steps_to_success": mean([r["planning_steps_to_success"] for r in rows if r["success"]]),
        "mean_planning_latency_s": mean([r["planning_latency_total_s"] / r["planning_steps_executed"] for r in rows]),
        "modules": {},
    }
    for name in ("encoder", "predictor"):
        all_calls = []
        for row in rows:
            raw = json.loads((output / f"trial_{row['seed']:05d}" / "efficiency.json").read_text())
            all_calls.extend(c for c in raw["calls"] if c["module"] == name)
        summary["modules"][name] = module_metrics(all_calls)
    write_json(output / "summary.json", summary)
    if rows:
        flat = [flatten(r) for r in rows]
        keys = list(dict.fromkeys(k for row in flat for k in row))
        with (output / "trials.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(flat)
    print(f"Completed {len(rows)}/{config['trials']}; SR={summary['success_rate']}; progress={summary['mean_task_progress']}", flush=True)
    return summary
