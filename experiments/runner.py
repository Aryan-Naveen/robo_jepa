# Copyright (c) 2026 RoboJEPA experiment contributors.
# Licensed under the LICENSE file in the root directory of this source tree.

"""Portable single-GPU experiment launcher; no shell rewriting or machine paths."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STEPS = {"22m": "150k", "50m": "120k", "100m": "150k", "300m": "300k", "1b": "300k", "2b": "450k", "4b": "450k", "8b": "300k"}
TASKS = ("reach", "object_reach", "obstacle_reach", "push")
# Presets fill only the options left unset on the command line.
#   smoke: check that runs, plots and videos work end to end (~10 min for 100M on one GPU)
#   full:  the candidates-per-iteration ladder at the paper's settings
PRESETS = {
    "smoke": {"samples": [16, 64, 256], "trials": 1, "planning_steps": 6, "cem_iterations": 3, "later_cem_iterations": 2},
    "full": {"samples": [16, 32, 64, 128, 256, 512, 1024], "trials": 10, "cem_iterations": 15, "later_cem_iterations": 10},
}
DEFAULTS = {"samples": [1024], "trials": 3, "cem_iterations": 15, "later_cem_iterations": 10}


def elites(samples, fraction, minimum):
    """CEM elites kept per iteration: a fixed fraction of the candidates (0.02 x 1024 = 20, the upstream setting),
    at least `minimum` so the refit std is not degenerate, and at most half the candidates."""
    return min(max(2, samples // 2), max(minimum, round(fraction * samples)))


def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run frozen-model RoboCasa trials")
    r.add_argument("--task", choices=TASKS, default="obstacle_reach")
    r.add_argument("--model-size", choices=STEPS, default="300m")
    r.add_argument("--preset", choices=PRESETS, help="smoke: quick end-to-end check; full: the samples ladder")
    r.add_argument("--trials", type=positive, help="episodes per candidate count (default 3)")
    r.add_argument("--seed", type=int, default=0, help="first task/planner seed; incremented for each trial")
    r.add_argument("--scene", type=int, help="fixed kitchen seed (push: kitchen ID); default 0, push 4")
    r.add_argument("--models-dir", type=Path, default=Path("models"))
    r.add_argument("--checkpoint-step", help="e.g. 300k; defaults to the released final checkpoint")
    r.add_argument("--checkpoint-dir", type=Path, help="explicit directory with config.json, world_model.pth.tar, encoder.pth.tar")
    r.add_argument("--output", type=Path, required=True)
    r.add_argument("--samples", type=positive, nargs="+",
                   help="total CEM candidates per iteration (default 1024); several values run a ladder, one "
                        "sub-directory per value, with the same episodes (seeds) for every value")
    r.add_argument("--batch-size", type=positive, default=16, help="candidates evaluated together; use 8 for 2B on 32 GiB")
    r.add_argument("--planning-steps", type=positive, help="execution budget (18 for obstacle/push; 5 otherwise)")
    r.add_argument("--horizon", type=positive, help="rollout length (5 for obstacle/push; 1 otherwise)")
    r.add_argument("--cem-iterations", type=positive, help="CEM iterations for the first decision (default 15)")
    r.add_argument("--later-cem-iterations", type=positive, help="CEM iterations for later decisions (default 10)")
    r.add_argument("--elite-fraction", type=float, default=0.02, help="CEM elites as a fraction of the candidates")
    r.add_argument("--min-elites", type=positive, default=4)
    r.add_argument("--success-distance", type=float, default=0.05,
                   help="reach tasks: success = final end-effector distance to the goal <= this (metres; provisional)")
    r.add_argument("--success-reference", choices=("goal_pose", "target"), default="goal_pose",
                   help="goal_pose: where the goal image was taken (what the planner sees); target: the env's "
                        "sampled target point (its reward), which the goal image may not reach")
    r.add_argument("--object", help="object_reach: banana; push: flourbag001")
    r.add_argument("--reach-min-distance", type=float, default=0.25, help="ObstacleReach start–goal minimum, metres")
    r.add_argument("--reach-max-distance", type=float, default=0.50, help="ObstacleReach start–goal maximum, metres")
    r.add_argument("--obstacle-clearance", type=float, default=0.03, help="metres")
    r.add_argument("--gpu", default="0")
    r.add_argument("--port", type=int, default=7300)
    r.add_argument("--master-port", type=int, default=29531)
    r.add_argument("--startup-timeout", type=positive, default=600)
    r.add_argument("--trial-timeout", type=positive, default=14400)
    r.add_argument("--flops", choices=("off", "once", "all"), default="once",
                   help="once: count each input shape's FLOPs once and reuse (default); all: count every forward (slow)")
    r.add_argument("--fast-planner", action=argparse.BooleanOptionalAction, default=True,
                   help="skip the planner's per-batch gc.collect() and GPU->CPU state copy (same numerics)")
    r.add_argument("--report", action=argparse.BooleanOptionalAction, default=True,
                   help="refresh <output>/report after every trial")
    r.add_argument("--resume", action="store_true", help="reuse this exact configuration and skip completed trials")
    r.add_argument("--dry-run", action="store_true", help="print resolved configuration and commands without running")
    s = sub.add_parser("summarize", help="rebuild aggregate JSON/CSV from completed trials")
    s.add_argument("output", type=Path)
    rep = sub.add_parser("report", help="tables, plots and per-episode videos for a run or a ladder")
    rep.add_argument("output", type=Path)
    rep.add_argument("--no-videos", action="store_true", help="skip the per-episode videos (faster)")
    rep.add_argument("--force", action="store_true", help="re-render per-episode trajectories and videos")
    rep.add_argument("--compare", type=Path, nargs="*", default=None,
                     help="other runs or ladders (e.g. another model size, same seeds) shown alongside; saved for "
                          "later refreshes; pass with no value to clear")
    return p


def fill_defaults(a):
    for source in (PRESETS.get(a.preset, {}), DEFAULTS):
        for key, value in source.items():
            if getattr(a, key, None) is None:
                setattr(a, key, value)
    return a


def ladder(a):
    """One resolved configuration per candidate count."""
    fill_defaults(a)
    configs = []
    for n in a.samples:
        b = argparse.Namespace(**vars(a))
        b.samples = n
        if len(a.samples) > 1:
            b.output = a.output / f"n{n:04d}"
        configs.append(resolve(b))
    return configs


def resolve(a):
    c = vars(a).copy()
    for k in ("command", "resume", "dry_run", "preset", "report"):
        c.pop(k, None)
    if not 0 < a.elite_fraction < 1:
        raise ValueError("--elite-fraction must be in (0, 1)")
    c["elites"] = elites(a.samples, a.elite_fraction, a.min_elites)
    c["scene"] = a.scene if a.scene is not None else (4 if a.task == "push" else 0)
    c["planning_steps"] = a.planning_steps or (18 if a.task in ("obstacle_reach", "push") else 5)
    c["horizon"] = a.horizon or (5 if a.task in ("obstacle_reach", "push") else 1)
    c["object"] = a.object or ("flourbag001" if a.task == "push" else "banana")
    c["batch_size"] = min(a.batch_size, a.samples)
    if a.samples < 2:
        raise ValueError("--samples must be at least 2 for CEM elite variance")
    if a.samples % c["batch_size"]:
        raise ValueError("--samples must be divisible by --batch-size")
    if a.seed < 0 or c["scene"] < 0:
        raise ValueError("seeds must be nonnegative")
    if not 0 <= a.reach_min_distance <= a.reach_max_distance or a.obstacle_clearance < 0:
        raise ValueError("invalid distance/clearance range")
    if a.port == a.master_port or not all(0 < p < 65536 for p in (a.port, a.master_port)):
        raise ValueError("choose distinct valid server and distributed ports")
    c["checkpoint_step"] = a.checkpoint_step or STEPS[a.model_size]
    for k in ("models_dir", "output", "checkpoint_dir"):
        c[k] = str(c[k].expanduser().resolve()) if c[k] is not None else None
    c["embodiment_id"] = 3
    c["schema_version"] = 2
    return c


def commands(c, trial):
    server = [sys.executable, str(ROOT / "scripts/deployment/robots/robocasa/robot_env_server.py"),
              "--host", "127.0.0.1", "--port", str(c["port"]), "--camera_names", "robot0_agentview_left", "robot0_eye_in_hand",
              "--camera_width", "320", "--camera_height", "240", "--robot", "PandaOmron", "--chunk_size", "4",
              "--wm_embodiment", "franka_omron_state_delta", "--seed", str(c["scene"] if c["task"] != "push" else 0),
              "--blocking_control", "--blocking_pos_tol", "0.001", "--blocking_max_steps", "200"]
    if c["task"] == "push":
        server += ["--object_push_task", "--object_push_target_kitchen_id", str(c["scene"]),
                   "--object_push_object_name", c["object"], "--object_push_forward_cone_deg", "60", "--object_push_clear_radius", "1.0"]
    elif c["task"] == "object_reach":
        server += ["--task_name", "ObjectReachKitchen", "--object_reach_task", "--object_reach_object_name", c["object"],
                   "--object_reach_task_seed", str(c["scene"]), "--object_reach_max_distance", "0.15",
                   "--object_reach_clear_radius", "1.0", "--object_reach_max_retries", "3"]
    else:
        server += ["--task_name", "CloseStandMixerHead", "--reach_task_seed", str(c["scene"]), "--reach_clear_radius", "1.0"]
        if c["task"] == "obstacle_reach":
            server += ["--obstacle_reach_task", "--reach_min_distance", str(c["reach_min_distance"]),
                       "--reach_max_distance", str(c["reach_max_distance"]), "--obstacle_clearance_m", str(c["obstacle_clearance"]),
                       "--kitchen_split", "target", "--obstacle_classes", "CoffeeMachine"]
        else:
            server += ["--reach_task", "--reach_max_distance", "0.30"]
    planner = [sys.executable, "-m", "experiments.instrument", "--checkpoint_path", str(Path(c["output"]) / "checkpoint"),
               "--host", "localhost", "--port", str(c["port"]), "--embed_dim", "1664", "--image_size", "240", "320",
               "--use_cameras", "0", "1", "--camera_id", "0", "2", "--embodiment_id", "3",
               "--task_path", str(trial / "tasks"), "--episodes", "1", "--init_ep", "0",
               "--total_steps", str(c["planning_steps"]), "--rollout", str(c["horizon"]),
               "--cem", str(c["cem_iterations"]), "--step_cem", str(c["later_cem_iterations"]),
               "--cutoff", str(c["elites"]),  # > 1: an absolute elite count in mpc_utils.cem
               "--samples", str(c["batch_size"]), "--num_batches", str(c["samples"] // c["batch_size"]),
               "--maxnorm", "0.3", "--maxgripper", "0.99", "--robot_init_std", "0.3", "--gripper_init_std", "0.8",
               "--momentum_std", "0.70", "--momentum_mean_gripper", "0.15", "--momentum_std_gripper", "0.45",
               "--execution_prefix", "1", "--closed_loop_context_len", "2", "--gamma", "0.99",
               "--translation_only", "--use_cache", "--load_compiled", "false", "--action_substeps", "1",
               "--bypass_input", "--view_weights", "0.5", "0.5", "--output_filename", str(trial / "episode")]
    return server, planner


def task_spec(c, seed):
    if c["task"] == "push":
        data = {"object_push_task_seed": seed}
    elif c["task"] == "object_reach":
        data = {"seed": c["scene"], "object_reach_task_seed": seed, "object_reach_object_name": c["object"]}
    else:
        data = {"seed": c["scene"], "reach_task_seed": seed}
    return {mode: dict(data, mode=mode) for mode in ("init", "goal")}


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def stage_checkpoint(c):
    source = Path(c["checkpoint_dir"]) if c["checkpoint_dir"] else Path(c["models_dir"]) / f'wm-scaling-{c["model_size"]}-norms'
    weight = "world_model.pth.tar" if c["checkpoint_dir"] else f'cooldown_{c["model_size"]}_{c["checkpoint_step"]}.pth.tar'
    files = {"config.json": source / "config.json", "encoder.pth.tar": source / "encoder.pth.tar", "world_model.pth.tar": source / weight}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {path}. See experiments/README.md.")
    target = Path(c["output"]) / "checkpoint"
    target.mkdir(exist_ok=True)
    provenance = {}
    for name, source_path in files.items():
        dest = target / name
        if dest.is_symlink():
            dest.unlink()
        dest.symlink_to(source_path.resolve())
        provenance[name] = {"source": str(source_path.resolve()), "bytes": source_path.stat().st_size,
                            "mtime_ns": source_path.stat().st_mtime_ns}
    # Preserve exact model architecture config alongside the source paths.
    provenance["model_config"] = json.loads(files["config.json"].read_text())
    return provenance


def stop(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def run_trial(c, seed):
    trial = Path(c["output"]) / f"trial_{seed:05d}"
    trial.mkdir(exist_ok=True)
    (trial / "tasks").mkdir(exist_ok=True)
    write_json(trial / "tasks/task.json", task_spec(c, seed))
    server_cmd, planner_cmd = commands(c, trial)
    write_json(trial / "commands.json", {"server": server_cmd, "planner": planner_cmd})
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "scripts/deployment")]),
               PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl",
               CUDA_VISIBLE_DEVICES=c["gpu"], LOCAL_RANK="0", RANK="0", WORLD_SIZE="1", MASTER_ADDR="localhost",
               MASTER_PORT=str(c["master_port"]), EXPERIMENT_TRIAL=str(trial), EXPERIMENT_SEED=str(seed),
               EXPERIMENT_FLOPS=c["flops"], EXPERIMENT_FAST_PLANNER=str(int(c["fast_planner"])))
    # Fail before spawning if another experiment owns either port.
    for port in (c["port"], c["master_port"]):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    server = planner = None
    started = time.monotonic()
    try:
        with (trial / "server.log").open("w") as log:
            server = subprocess.Popen(server_cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while "[GymEnvServer] listening" not in (trial / "server.log").read_text(errors="replace"):
                if server.poll() is not None:
                    raise RuntimeError(f"Environment exited ({server.returncode}); see {trial / 'server.log'}")
                if time.monotonic() - started > c["startup_timeout"]:
                    raise TimeoutError("Environment startup timed out")
                time.sleep(0.5)
            with (trial / "planner.log").open("w") as plog:
                planner = subprocess.Popen(planner_cmd, cwd=ROOT, env=env, stdout=plog, stderr=subprocess.STDOUT, start_new_session=True)
                planner.wait(timeout=c["trial_timeout"])
                if planner.returncode:
                    raise RuntimeError(f"Planner exited ({planner.returncode}); see {trial / 'planner.log'}")
        from .metrics import trial_metrics
        metrics = trial_metrics(trial, c)
        metrics["trial_wall_s"] = time.monotonic() - started
        write_json(trial / "metrics.json", metrics)
        (trial / "failure.json").unlink(missing_ok=True)
    except BaseException as exc:
        write_json(trial / "failure.json", {"error": str(exc), "type": type(exc).__name__})
        raise
    finally:
        stop(planner)
        stop(server)


def prepare(c, resume):
    """Create (or, with --resume, validate) one run directory and write its config and provenance."""
    out = Path(c["output"])
    if out.exists() and any(out.iterdir()):
        if not resume or not (out / "config.json").exists():
            raise ValueError(f"{out} is not empty; use a new directory or --resume")
        if json.loads((out / "config.json").read_text()) != c:
            raise ValueError(f"--resume requires the identical experiment configuration ({out})")
    out.mkdir(parents=True, exist_ok=True)
    provenance = stage_checkpoint(c)
    if resume and (out / "checkpoint_provenance.json").exists():
        if json.loads((out / "checkpoint_provenance.json").read_text()) != provenance:
            raise ValueError("checkpoint files changed since the original run")
    write_json(out / "config.json", c)
    write_json(out / "checkpoint_provenance.json", provenance)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    if not (out / "revision.json").exists():
        write_json(out / "revision.json", {"commit": git.stdout.strip(), "python": sys.version,
            "dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout)})


def refresh_report(root):
    """Rebuild the report from the finished episodes; a plotting problem must never stop the experiment."""
    try:
        from .report import report
        report(root, quiet=True)
        print(f"report updated: {root / 'report' / 'README.md'}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"report refresh failed (run continues): {exc}", flush=True)


def main():
    p = parser()
    a = p.parse_args()
    if a.command == "summarize":
        from .metrics import summarize
        summarize(a.output.resolve())
        return
    if a.command == "report":
        from .report import report
        report(a.output.resolve(), videos=not a.no_videos, force=a.force, compare=a.compare)
        return
    try:
        configs = ladder(a)
        if a.dry_run:
            for c in configs:
                server, planner = commands(c, Path(c["output"]) / f'trial_{c["seed"]:05d}')
                print(json.dumps({"config": c, "server": server, "planner": planner}, indent=2))
            return
        for c in configs:
            prepare(c, a.resume)
        if len(configs) > 1:
            # Merge with an existing ladder so more N can be added to the same directory later.
            path = a.output.resolve() / "ladder.json"
            known = dict(zip(*(json.loads(path.read_text())[k] for k in ("samples", "runs")))) if path.exists() else {}
            known.update({c["samples"]: Path(c["output"]).name for c in configs})
            write_json(path, {"runs": [known[n] for n in sorted(known)], "samples": sorted(known)})
        from .metrics import summarize
        # Seeds outside, candidate counts inside: a partial ladder still covers every count.
        first = configs[0]
        for seed in range(first["seed"], first["seed"] + first["trials"]):
            for c in configs:
                out = Path(c["output"])
                if a.resume and (out / f"trial_{seed:05d}/metrics.json").exists():
                    print(f"skip completed trial {seed} (N={c['samples']})", flush=True)
                    continue
                print(f"Running {c['task']} / {c['model_size']} / N={c['samples']}, trial seed {seed}", flush=True)
                try:
                    run_trial(c, seed)
                except (RuntimeError, TimeoutError, subprocess.TimeoutExpired, ValueError) as exc:
                    # Recorded in failure.json and counted by summarize; keep going with the rest of the ladder.
                    print(f"trial {seed} (N={c['samples']}) failed: {exc}", flush=True)
                finally:
                    summarize(out)
                if a.report:
                    refresh_report(a.output.resolve())
    except (ValueError, FileNotFoundError, RuntimeError, TimeoutError, subprocess.TimeoutExpired, OSError) as exc:
        p.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
