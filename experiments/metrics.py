# Copyright (c) 2026 RoboJEPA experiment contributors.
# Licensed under the LICENSE file in the root directory of this source tree.

"""Explicit task and efficiency metric definitions shared by runs and summaries."""
import csv
import json
from pathlib import Path
import statistics

import numpy as np

from .runner import write_json

REACH_TASKS = ("reach", "object_reach", "obstacle_reach")


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def module_metrics(calls):
    if not calls:
        return None
    # Forwards run under FlopCounterMode are slower; time from the others and scale to all calls.
    timed = [c for c in calls if not c.get("flops_counted")] or calls
    latency = sum(c["latency_s"] for c in timed) * len(calls) / len(timed)
    samples = sum(c["batch_size"] for c in calls)
    tokens = sum(c["output_tokens"] for c in calls)
    measured_flops = [c["flops"] for c in calls]
    flops = sum(measured_flops) if all(f is not None for f in measured_flops) else None
    peaks = [c["peak_allocated_bytes"] for c in calls if c["peak_allocated_bytes"] is not None]
    increments = [c["peak_increment_bytes"] for c in calls if c["peak_increment_bytes"] is not None]
    # Boundary average, explicitly NOT a continuous time-weighted memory estimate.
    memory = [v for c in calls for v in (c["allocated_before_bytes"], c["allocated_after_bytes"]) if v is not None]
    return {
        "forward_calls": len(calls), "timed_calls": len(timed), "sample_forwards": samples, "output_tokens": tokens,
        "tokens_per_second": tokens / latency if latency else None,
        "latency_total_s": latency, "latency_mean_batch_s": latency / len(calls),
        "latency_amortized_sample_s": latency / samples,
        "flops": flops, "macs": flops / 2 if flops is not None else None,
        "average_boundary_allocated_bytes": mean(memory),
        "peak_allocated_bytes": max(peaks) if peaks else None,
        "peak_increment_bytes": max(increments) if increments else None,
    }


def goal_gap(trial, poses):
    """Distance between the goal-image end-effector pose and the env's target point (needs scene.json)."""
    scene = trial / "scene.json"
    if poses is None or not scene.exists():
        return None
    goal = json.loads(scene.read_text()).get("init", {}).get("reach_goal_world")
    return float(np.linalg.norm(poses[0] - np.asarray(goal))) if goal else None


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
        env_success = np.asarray(data["env_info_success"], dtype=bool)
        if len(env_success) != len(rewards):
            raise ValueError("Success flags and reward steps are misaligned")
        decisions = efficiency["decisions"]
        # execution_prefix=action_substeps=1 is fixed by the public runner:
        # one logged environment action corresponds to one planning decision.
        if len(decisions) != len(rewards):
            raise ValueError("Planning decisions and environment steps differ; cannot map steps-to-success")
        distance_keys = [k for k in data.files if k.endswith("distance_to_goal")]
        key = next((k for k in distance_keys if "push" in k), distance_keys[0] if distance_keys else None)
        target_distances = np.asarray(data[key], dtype=float) if key else None
        d0 = next((k for k in ("env_info_reach_d0", "reach_d0") if k in data.files), None)
        target_start = float(np.asarray(data[d0]).ravel()[0]) if d0 else None
        poses = np.asarray(data["poses"], dtype=float)[:, :3] if "poses" in data.files else None
    # Two references for "distance to the goal":
    #  goal_pose: where the end effector was when the goal image was rendered (poses[0]). This is what the planner
    #             is asked to reach, since it only sees the goal image.
    #  target:    the env's sampled target point (reach_goal_world), used by the env's reward. The env's expert
    #             can stop short of it when rendering the goal (e.g. 11.7 cm in scene 0 seed 0), so even a perfect
    #             image match can score far from 0.
    pose_distances = pose_start = None
    if poses is not None and len(poses) == len(rewards) + 2:  # [goal, start, after step 1..T]
        pose_distances = np.linalg.norm(poses[2:] - poses[0], axis=1)
        pose_start = float(np.linalg.norm(poses[1] - poses[0]))
    reference = config.get("success_reference", "goal_pose")
    if reference == "goal_pose" and pose_distances is not None:
        distances, start_distance = pose_distances, pose_start
    else:
        reference = "target" if target_distances is not None else None
        distances, start_distance = target_distances, target_start
    final_distance = float(distances[-1]) if distances is not None else None
    if config["task"] in REACH_TASKS and distances is not None:
        # The env's success flag belongs to the host RoboCasa task the reach tasks reuse as a backdrop
        # (CloseStandMixerHead), so it is always False here. Success = end effector within the threshold.
        successes = distances <= config.get("success_distance", 0.05)
    else:
        successes = env_success
    indices = np.flatnonzero(successes)
    first_success = int(indices[0]) + 1 if len(indices) else None
    module_stats = {name: module_metrics([c for c in efficiency["calls"] if c["module"] == name]) for name in ("encoder", "predictor")}
    per_decision = []
    for d in decisions:
        calls = [c for c in efficiency["calls"] if c["decision"] == d["index"] and c["module"] == "predictor"]
        per_decision.append({**d, "predictor": module_metrics(calls)})
    return {
        "seed": int(trial.name.split("_")[-1]), "task": config["task"], "model_size": config["model_size"],
        "samples": config["samples"], "elites": config.get("elites"),
        "success": bool(successes[-1]) if config["task"] in REACH_TASKS else bool(len(indices)), "success_any_step": bool(len(indices)), "env_success_any_step": bool(env_success.any()),
        "planning_steps_to_success": first_success,
        "planning_steps_executed": len(decisions),
        "task_progress": float(rewards.mean() if config["task"] in ("reach", "object_reach") else rewards[-1]),
        "final_progress": float(rewards[-1]), "mean_progress": float(rewards.mean()),
        "progress_by_step": rewards.tolist(), "final_distance_m": final_distance, "start_distance_m": start_distance,
        "distance_by_step": distances.tolist() if distances is not None else None, "distance_reference": reference,
        "final_distance_goal_pose_m": float(pose_distances[-1]) if pose_distances is not None else None,
        "final_distance_target_m": float(target_distances[-1]) if target_distances is not None else None,
        "goal_pose_to_target_m": goal_gap(trial, poses),
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
