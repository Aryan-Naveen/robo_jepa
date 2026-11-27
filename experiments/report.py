# Copyright (c) 2026 RoboJEPA experiment contributors.
# Licensed under the LICENSE file in the root directory of this source tree.

"""Human-readable report for one run or a candidates-per-iteration ladder.

    python -m experiments report results/<run-or-ladder> [--compare results/<other run or ladder> ...]

Works on partial results (only finished episodes are read), and `run` refreshes it after every trial. Per-episode
trajectory and video files are rendered once; `--force` re-renders them. `--compare` adds other runs (e.g. another
model size on the same seeds) as extra rows, points and columns; the list is saved in report/report.json so the
automatic refresh during `run` keeps it.

writes <dir>/report/:
    README.md                 summary table (one row per candidate count) and links to everything below
    summary.csv, episodes.csv machine-readable tables
    time_vs_samples.png       planning time per decision vs candidates per CEM iteration
    success_vs_samples.png    success rate and final distance vs candidates
    success_vs_time.png       success rate and final distance vs planning time (where faster models shift left)
    distance_vs_step.png      distance to goal over the episode, one line per candidate count
    trajectories.png          top-down end-effector paths with the obstacle, episodes x candidate counts
    episodes/<run>_seed<k>/   per-episode trajectory (top + side view) and a video: goal | cameras | path
"""
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

# Reference palette (dataviz skill, light surface). Candidate counts are ordered magnitudes, so they use one
# sequential blue ramp, light -> dark; categorical slots mark start / goal / obstacle.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
OBSTACLE = "#9a9890"
VIOLET = "#6250d6"  # comparison runs (e.g. another model size)
EXTRA_DISTANCES_M = (0.10, 0.15)  # looser secondary success thresholds, shown next to the primary --success-distance
THRESHOLD_COLORS = [RAMP[8], RAMP[4], RAMP[1]]  # primary (dark) -> loosest (light)
COMPARE_KEYS = ("task", "scene", "seed", "trials", "planning_steps", "horizon", "cem_iterations",
                "later_cem_iterations", "batch_size", "elite_fraction", "min_elites", "success_distance",
                "success_reference", "fast_planner")


def thresholds_m(primary):
    return [primary] + [d for d in EXTRA_DISTANCES_M if d > primary + 1e-9]


def tkey(d):
    return f"le{round(100 * d)}cm"


def success_series(row):
    """(threshold cm, count key, rate key, ci-low key, ci-high key) for the primary and each extra threshold."""
    out = [(100 * row["success_distance_m"], "successes", "success_rate", "success_ci_low", "success_ci_high")]
    for d in thresholds_m(row["success_distance_m"])[1:]:
        t = tkey(d)
        out.append((100 * d, f"successes_{t}", f"success_rate_{t}", f"success_{t}_ci_low", f"success_{t}_ci_high"))
    return out


def load_runs(root):
    root = Path(root)
    if (root / "ladder.json").exists():
        names = json.loads((root / "ladder.json").read_text())["runs"]
        dirs = [root / n for n in names]
    elif (root / "config.json").exists():
        dirs = [root]
    else:
        dirs = sorted(p.parent for p in root.glob("*/config.json"))
    runs = []
    for d in dirs:
        if not (d / "config.json").exists():
            continue
        config = json.loads((d / "config.json").read_text())
        episodes = []
        for t in sorted(d.glob("trial_*")):
            if (t / "metrics.json").exists():
                m = json.loads((t / "metrics.json").read_text())
                m["dir"] = t
                episodes.append(m)
        failed = sorted(int(t.name.split("_")[-1]) for t in d.glob("trial_*") if (t / "failure.json").exists()
                        and not (t / "metrics.json").exists())
        runs.append({"dir": d, "name": d.name, "config": config, "episodes": episodes, "failed": failed})
    runs.sort(key=lambda r: r["config"]["samples"])
    return runs


def wilson(k, n, z=1.96):
    """95% Wilson interval for a success rate (sane for tiny n and 0/n or n/n)."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def median(values):
    values = [v for v in values if v is not None]
    return float(np.median(values)) if values else math.nan


def mean_or_nan(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else math.nan


REFERENCE = {"goal_pose": "to the goal-image pose", "target": "to the env target"}


def ref_text(rows):
    return REFERENCE.get(rows[0].get("distance_reference"), "to the goal")


def per_decision_s(m):
    return m["planning_latency_total_s"] / max(1, m["planning_steps_executed"])


def predictor_s_per_decision(m):
    p = m["modules"]["predictor"]
    return p["latency_total_s"] / max(1, m["planning_steps_executed"]) if p else math.nan


def run_row(run):
    eps, c = run["episodes"], run["config"]
    n = len(eps)
    k = sum(e["success"] for e in eps)
    lo, hi = wilson(k, n)
    dist = np.array([e["final_distance_m"] for e in eps if e["final_distance_m"] is not None], dtype=float)
    extra = {}
    for d in thresholds_m(c.get("success_distance", 0.05))[1:]:
        k_d = int(sum(x <= d for x in dist))
        lo_d, hi_d = wilson(k_d, n)
        t = tkey(d)
        extra.update({f"successes_{t}": k_d, f"success_rate_{t}": k_d / n if n else math.nan,
                      f"success_{t}_ci_low": lo_d, f"success_{t}_ci_high": hi_d})
    plan = np.array([per_decision_s(e) for e in eps])
    pred = np.array([predictor_s_per_decision(e) for e in eps])
    calls = [e["modules"]["predictor"] for e in eps if e["modules"]["predictor"]]
    ms_call = 1e3 * sum(p["latency_total_s"] for p in calls) / max(1, sum(p["forward_calls"] for p in calls))
    flops = [p["flops"] for p in calls if p["flops"] is not None]
    steps = sum(e["planning_steps_executed"] for e in eps)
    return {
        "run": run["name"], "label": run.get("label", run["name"]), "model": c["model_size"],
        "compare": run.get("compare", False), "samples": c["samples"], "batch_size": c["batch_size"],
        "batches": c["samples"] // c["batch_size"], "elites": c.get("elites"), "episodes": n,
        "failed": len(run["failed"]), "successes": k, "success_rate": k / n if n else math.nan,
        "success_ci_low": lo, "success_ci_high": hi, "success_distance_m": c.get("success_distance", 0.05),
        **extra,
        "final_distance_mean_m": float(dist.mean()) if len(dist) else math.nan,
        "final_distance_sd_m": float(dist.std(ddof=1)) if len(dist) > 1 else math.nan,
        "final_distance_median_m": float(np.median(dist)) if len(dist) else math.nan,
        "final_distance_q1_m": float(np.percentile(dist, 25)) if len(dist) else math.nan,
        "final_distance_q3_m": float(np.percentile(dist, 75)) if len(dist) else math.nan,
        "final_progress_mean": float(np.mean([e["final_progress"] for e in eps])) if n else math.nan,
        "final_distance_target_median_m": median([e.get("final_distance_target_m") for e in eps]),
        "goal_pose_to_target_mean_m": mean_or_nan([e.get("goal_pose_to_target_m") for e in eps]),
        "distance_reference": eps[0].get("distance_reference", "target") if n else None,
        "planning_s_per_decision": float(plan.mean()) if n else math.nan,
        "predictor_s_per_decision": float(pred.mean()) if n else math.nan,
        "predictor_share": float(pred.sum() / plan.sum()) if n and plan.sum() else math.nan,
        "predictor_ms_per_call": ms_call if calls else math.nan,
        "predictor_tflops_per_decision": sum(flops) / steps / 1e12 if flops and len(flops) == len(calls) else math.nan,
        "episode_wall_min": float(np.mean([e.get("trial_wall_s", math.nan) for e in eps])) / 60 if n else math.nan,
    }


# ---------------------------------------------------------------- plotting helpers

def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 9.5,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8.5, "font.size": 9.5,
        "lines.linewidth": 2, "lines.markersize": 6,
    })
    return plt


def colors_for(values):
    if len(values) == 1:
        return {values[0]: BLUE}
    idx = np.linspace(0, len(RAMP) - 1, len(values)).round().astype(int)
    return {v: RAMP[i] for v, i in zip(values, idx)}


def log2_axis(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()


def subtitle(fig, text):
    fig.text(0.01, 0.005, text, fontsize=8, color=INK2, ha="left", va="bottom")


def setting_text(run):
    c = run["config"]
    return (f"{c['task']} · {c['model_size']} · {c['planning_steps']} steps · horizon {c['horizon']} · "
            f"CEM {c['cem_iterations']}+{c['later_cem_iterations']} iterations · batch {c['batch_size']} · "
            f"{'fast planner' if c.get('fast_planner', False) else 'stock planner'} · FLOPs {c.get('flops', 'all')}")


# ---------------------------------------------------------------- ladder plots

def plot_time(plt, runs, rows, out, cmp_rows=()):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    xs = [r["samples"] for r in rows]
    for run in runs:
        n = run["config"]["samples"]
        ax.scatter([n] * len(run["episodes"]), [per_decision_s(e) for e in run["episodes"]], s=14, color=BLUE,
                   alpha=0.35, linewidths=0, zorder=2)
    ax.plot(xs, [r["planning_s_per_decision"] for r in rows], "-o", color=BLUE, label="planning (all)", zorder=3)
    ax.plot(xs, [r["predictor_s_per_decision"] for r in rows], "-o", color=ORANGE, label="predictor forwards only",
            zorder=3)
    if len(xs) > 1 and rows[-1]["planning_s_per_decision"] > 0:
        # Reference: time exactly proportional to N, anchored at the largest N.
        ref = rows[-1]["planning_s_per_decision"]
        ax.plot(xs, [ref * x / xs[-1] for x in xs], "--", color=INK2, linewidth=1, label="∝ N (reference)")
    for r in cmp_rows:
        ax.scatter(r["samples"], r["planning_s_per_decision"], marker="D", s=46, color=VIOLET, zorder=4,
                   label=f"{r['label']} planning")
    log2_axis(ax, sorted({*xs, *(r["samples"] for r in cmp_rows)}))
    ax.set_yscale("log")
    ax.set_xlabel("candidate action sequences per CEM iteration (N)")
    ax.set_ylabel("seconds per decision (log)")
    ax.set_title("Does more candidates mean more time?")
    ax.legend(loc="upper left")
    subtitle(fig, setting_text(runs[0]) + "  ·  dots: episodes, line: mean")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "time_vs_samples.png", dpi=150)
    plt.close(fig)


def success_panels(plt, runs, rows, xkey, xlabel, title, path, log_x, cmp_runs=(), cmp_rows=()):
    from matplotlib.lines import Line2D
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.4))
    xs = [r[xkey] for r in rows]
    series = success_series(rows[0])
    for i in reversed(range(len(series))):  # loosest first, primary drawn on top
        cm, k, rate, lo_k, hi_k = series[i]
        color = THRESHOLD_COLORS[i]
        a1.fill_between(xs, [r[lo_k] for r in rows], [r[hi_k] for r in rows], color=color, alpha=0.08, linewidth=0)
        a1.plot(xs, [r[rate] for r in rows], "-o", color=color, label=f"final distance ≤ {cm:.0f} cm",
                zorder=3 + len(series) - i, markersize=4 + 1.5 * i)
        for x, r in zip(xs, rows):
            if i and r[k] == r[series[i - 1][1]]:
                continue  # same count as the next stricter threshold: one label is enough
            a1.annotate(f"{r[k]}/{r['episodes']}", (x, r[rate]), textcoords="offset points",
                        xytext=(0, 7 if i == 0 else -12), ha="center", fontsize=8, color=INK2)
    for r in cmp_rows:
        x = r[xkey]
        for i, (cm, k, rate, _, _) in enumerate(series):
            a1.scatter(x, r[rate], marker="D", s=48, facecolor=THRESHOLD_COLORS[i], edgecolor=VIOLET,
                       linewidths=1.8, zorder=8)
            if i and r[k] == r[series[i - 1][1]]:
                continue
            a1.annotate(f"{r[k]}/{r['episodes']}", (x, r[rate]), textcoords="offset points", xytext=(8, -3),
                        ha="left", fontsize=8, color=VIOLET)
    handles, labels = a1.get_legend_handles_labels()
    handles, labels = handles[::-1], labels[::-1]
    for r in cmp_rows:
        handles.append(Line2D([], [], marker="D", linestyle="none", markerfacecolor="white", markeredgecolor=VIOLET,
                              markeredgewidth=1.8, markersize=7))
        labels.append(f"{r['label']} (fill = threshold)")
    a1.legend(handles, labels, loc="upper left")
    a1.set_ylim(-0.03, 1.08)
    a1.set_ylabel("success rate")
    fig.text(0.01, 0.955, f"distance measured {ref_text(rows)}", fontsize=8, color=INK2)
    a1.set_title(title)
    # Final distance: one thin gray line per episode seed (paired across N), dots, median and mean.
    by_seed = {}
    for run, x in zip(runs, xs):
        for e in run["episodes"]:
            by_seed.setdefault(e["seed"], []).append((x, 100 * e["final_distance_m"]))
            a2.scatter(x, 100 * e["final_distance_m"], s=16, color=BLUE, alpha=0.5, linewidths=0, zorder=3)
    for pts in by_seed.values():
        pts.sort()
        a2.plot(*zip(*pts), color=GRID, linewidth=1, zorder=1)
    a2.plot(xs, [100 * r["final_distance_median_m"] for r in rows], "-o", color=INK, linewidth=1.5,
            markersize=4, label="median", zorder=4)
    a2.plot(xs, [100 * r["final_distance_mean_m"] for r in rows], "--s", color=INK2, linewidth=1.5,
            markersize=4, label="mean", zorder=4)
    for run, r in zip(cmp_runs, cmp_rows):
        x = r[xkey]
        a2.scatter([x] * len(run["episodes"]), [100 * e["final_distance_m"] for e in run["episodes"]], s=16,
                   color=VIOLET, alpha=0.6, linewidths=0, zorder=5)
        a2.scatter(x, 100 * r["final_distance_median_m"], marker="D", s=48, color=VIOLET, zorder=6,
                   label=f"{r['label']} median")
        a2.scatter(x, 100 * r["final_distance_mean_m"], marker="s", s=44, facecolor="white", edgecolor=VIOLET,
                   linewidths=1.8, zorder=6, label=f"{r['label']} mean")
    for i, (cm, *_) in enumerate(series):
        a2.axhline(cm, color=THRESHOLD_COLORS[i], linestyle=":", linewidth=1.2, label=f"{cm:.0f} cm")
    a2.set_ylim(bottom=0)
    a2.set_ylabel(f"final distance {ref_text(rows)} (cm)")
    a2.set_title("Final distance (gray lines join the same episode)")
    a2.legend(loc="upper right", fontsize=7.5)
    for ax in (a1, a2):
        if log_x:
            log2_axis(ax, sorted({*xs, *(r[xkey] for r in cmp_rows)}))
        ax.set_xlabel(xlabel)
    subtitle(fig, setting_text(runs[0]) + "  ·  bands: 95% Wilson interval")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_distance_steps(plt, runs, out, cmp_runs=()):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cols = colors_for([r["config"]["samples"] for r in runs])
    for run in [*runs, *cmp_runs]:
        curves = [e["distance_by_step"] for e in run["episodes"] if e.get("distance_by_step")]
        if not curves:
            continue
        L = min(len(c) for c in curves)
        arr = 100 * np.array([c[:L] for c in curves])
        start = [100 * e["start_distance_m"] for e in run["episodes"] if e.get("start_distance_m")]
        x = np.arange(0, L + 1)
        y = np.concatenate([[np.mean(start) if start else np.nan], arr.mean(0)])
        if run.get("compare"):
            ax.plot(x, y, "--", color=VIOLET, label=run["label"])
        else:
            ax.plot(x, y, "-", color=cols[run["config"]["samples"]], label=f"N = {run['config']['samples']}")
    for d, style in zip(thresholds_m(runs[0]["config"].get("success_distance", 0.05)), ["--", ":", (0, (1, 4))]):
        ax.axhline(100 * d, color=AQUA, linestyle=style, linewidth=1.2)
        ax.annotate(f"{100 * d:.0f} cm", (1, 100 * d), xycoords=("axes fraction", "data"), xytext=(-2, 2),
                    textcoords="offset points", ha="right", fontsize=7.5, color=INK2)
    ax.set_xlabel("executed step")
    ax.set_ylabel(f"mean distance {REFERENCE.get(runs[0]['episodes'][0].get('distance_reference'), 'to the goal')} (cm)")
    ax.set_ylim(bottom=0)
    ax.set_title("Distance to goal during the episode")
    ax.legend(loc="upper right", ncol=2)
    subtitle(fig, setting_text(runs[0]))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out / "distance_vs_step.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- scene geometry

def episode_geometry(e):
    """World-frame end-effector path, start, goal and obstacle footprint for one episode (None if unavailable)."""
    t = e["dir"]
    npz = next(t.glob("episode_*.npz"), None)
    if npz is None:
        return None
    with np.load(npz, allow_pickle=False) as d:
        poses = np.asarray(d["poses"], dtype=float)[:, :3]
    # poses[0]: goal reset (expert pose at the goal), poses[1]: start, poses[2:]: after each executed step.
    g = {"goal": poses[0], "path": poses[1:], "obstacle": None, "top_z": None, "target": None}
    scene_path = t / "scene.json"
    if scene_path.exists():
        scene = json.loads(scene_path.read_text())
        init = scene.get("init", {})
        if "reach_goal_world" in init:
            g["target"] = np.asarray(init["reach_goal_world"], dtype=float)
        if "reach_start_world" in init:
            g["start_world"] = np.asarray(init["reach_start_world"], dtype=float)
        keys = ("obstacle_x_base_m", "obstacle_x_half_extent_m", "obstacle_y_base_min", "obstacle_y_base_max")
        if all(k in init for k in keys) and "proprio_robocasa" in init:
            from scipy.spatial.transform import Rotation
            pr = np.asarray(init["proprio_robocasa"], dtype=float)
            base, R_wb = pr[:3], Rotation.from_quat(pr[3:7]).as_matrix()
            x0, hx = init["obstacle_x_base_m"], init["obstacle_x_half_extent_m"]
            y0, y1 = init["obstacle_y_base_min"], init["obstacle_y_base_max"]
            corners = np.array([[x0 - hx, y0, 0], [x0 + hx, y0, 0], [x0 + hx, y1, 0], [x0 - hx, y1, 0]])
            g["obstacle"] = base[None, :2] + (corners @ R_wb.T)[:, :2]
            g["top_z"] = init.get("obstacle_world_top_z")
            g["obstacle_class"] = init.get("obstacle_class", "obstacle")
    return g


def draw_topdown(ax, g, color=BLUE, upto=None, legend=False, compact=False):
    from matplotlib.patches import Polygon
    if g["obstacle"] is not None:
        ax.add_patch(Polygon(g["obstacle"], closed=True, facecolor=OBSTACLE, alpha=0.35, edgecolor=OBSTACLE,
                             linewidth=1, label=g.get("obstacle_class", "obstacle")))
    p = g["path"] if upto is None else g["path"][: upto + 1]
    ax.plot(p[:, 0], p[:, 1], "-", color=color, linewidth=1.5 if compact else 2, label="end effector", zorder=3)
    ax.scatter(p[1:, 0], p[1:, 1], s=6 if compact else 12, color=color, zorder=4, linewidths=0)
    ax.scatter(*g["path"][0, :2], s=30 if compact else 50, marker="o", color=SURFACE, edgecolors=INK, zorder=5,
               linewidths=1.2, label="start")
    ax.scatter(*g["goal"][:2], s=60 if compact else 110, marker="*", color=ORANGE, edgecolors=SURFACE, zorder=5,
               linewidths=0.8, label="goal (goal-image pose)")
    if g.get("target") is not None:
        ax.scatter(*g["target"][:2], s=25 if compact else 45, marker="x", color=INK2, zorder=5, linewidths=1.2,
                   label="env target")
    ax.set_aspect("equal", adjustable="box")
    if legend:
        ax.legend(loc="best", fontsize=8)


def frame_limits(geoms, pad=0.06):
    pts = []
    for g in geoms:
        pts += [g["path"][:, :2], g["goal"][None, :2]]
        if g.get("target") is not None:
            pts.append(g["target"][None, :2])
        if g["obstacle"] is not None:
            pts.append(g["obstacle"])
    pts = np.concatenate(pts)
    lo, hi = pts.min(0) - pad, pts.max(0) + pad
    # Square window so equal-aspect panels keep their size.
    centre, half = (lo + hi) / 2, (hi - lo).max() / 2
    return centre - half, centre + half


def plot_trajectory_grid(plt, runs, out):
    seeds = sorted({e["seed"] for r in runs for e in r["episodes"]})
    if not seeds:
        return False
    cols = colors_for([r["config"]["samples"] for r in runs if not r.get("compare")])
    fig, axes = plt.subplots(len(seeds), len(runs), figsize=(max(5.0, 2.3 * len(runs) + 0.6), 2.3 * len(seeds) + 1.0),
                             squeeze=False)
    for j, run in enumerate(runs):
        by_seed = {e["seed"]: e for e in run["episodes"]}
        for i, s in enumerate(seeds):
            ax = axes[i, j]
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for side in ax.spines.values():
                side.set_visible(True)
                side.set_color(GRID)
            if i == 0:
                ax.set_title(run["label"] if run.get("compare") else f"N = {run['config']['samples']}",
                             fontsize=9.5, loc="center", color=VIOLET if run.get("compare") else INK)
            if j == 0:
                ax.set_ylabel(f"episode {s}", fontsize=9)
            e = by_seed.get(s)
            g = episode_geometry(e) if e else None
            if g is None:
                ax.text(0.5, 0.5, "missing", transform=ax.transAxes, ha="center", color=INK2)
                continue
            draw_topdown(ax, g, color=VIOLET if run.get("compare") else cols[run["config"]["samples"]], compact=True)
            mark = "✓" if e["success"] else "✗"
            ax.text(0.03, 0.03, f"{mark} {100 * e['final_distance_m']:.1f} cm", transform=ax.transAxes, fontsize=8,
                    color=AQUA if e["success"] else RED, fontweight="bold")
    # Same zoom across a row (same episode = same scene).
    for i in range(len(seeds)):
        gs = [episode_geometry(e) for r in runs for e in r["episodes"] if e["seed"] == seeds[i]]
        gs = [g for g in gs if g is not None]
        if gs:
            lo, hi = frame_limits(gs)
            for ax in axes[i]:
                ax.set_xlim(lo[0], hi[0])
                ax.set_ylim(lo[1], hi[1])
    fig.suptitle("End-effector paths, top-down", x=0.01, y=1 - 0.12 / fig.get_figheight(), ha="left", va="top",
                 fontsize=11, fontweight="bold", color=INK)
    fig.text(0.01, 1 - 0.55 / fig.get_figheight(), "gray box: obstacle footprint · ○ start · ★ goal-image pose · × env target · same scene per row",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.75 / fig.get_figheight()))
    fig.savefig(out / "trajectories.png", dpi=130)
    plt.close(fig)
    return True


def plot_episode(plt, e, g, path):
    """Top-down and side view (height along the start->goal line) of one episode."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.6))
    draw_topdown(a1, g)
    lo, hi = frame_limits([g])
    a1.set_xlim(lo[0], hi[0])
    a1.set_ylim(lo[1], hi[1])
    a1.set_xlabel("world x (m)")
    a1.set_ylabel("world y (m)")
    a1.set_title("Top-down")
    draw_side(a2, g)
    (x0, x1), (z0, z1) = side_limits(g)
    a2.set_xlim(x0, x1)
    a2.set_ylim(z0, z1)
    a2.set_xlabel("distance along start → goal (m)")
    a2.set_ylabel("height z (m)")
    a2.set_title("Side view (obstacle box projected onto the start → goal line)")
    handles, labels = a2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8.5)
    mark = "success" if e["success"] else "fail"
    fig.suptitle(f"N = {e['samples']}, episode {e['seed']}: {mark}, final distance {100 * e['final_distance_m']:.1f} cm",
                 x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def read_frames(path):
    import imageio.v2 as imageio
    try:
        return [np.asarray(f) for f in imageio.mimread(path, memtest=False)]
    except Exception:
        return []


def draw_side(ax, g, upto=None, compact=False):
    """Height vs distance along the start->goal line, with the obstacle's extent along that line and its top."""
    start, goal = g["path"][0], g["goal"]
    u = (goal[:2] - start[:2]) / max(1e-9, np.linalg.norm(goal[:2] - start[:2]))
    path = g["path"] if upto is None else g["path"][: upto + 1]
    along = (path[:, :2] - start[:2]) @ u
    if g["obstacle"] is not None and g["top_z"] is not None:
        s = (g["obstacle"] - start[:2]) @ u
        floor = min(g["path"][:, 2].min(), goal[2], g["top_z"]) - 0.08
        ax.fill_between([s.min(), s.max()], [g["top_z"]] * 2, [floor] * 2, color=OBSTACLE, alpha=0.35, linewidth=0,
                        label=g.get("obstacle_class", "obstacle"))
    ax.plot(along, path[:, 2], "-", color=BLUE, linewidth=1.5 if compact else 2, label="end effector", zorder=3)
    ax.scatter(along[1:], path[1:, 2], s=6 if compact else 12, color=BLUE, zorder=4, linewidths=0)
    ax.scatter(0, start[2], s=30 if compact else 50, color=SURFACE, edgecolors=INK, zorder=5, linewidths=1.2,
               label="start")
    ax.scatter((goal[:2] - start[:2]) @ u, goal[2], s=60 if compact else 110, marker="*", color=ORANGE,
               edgecolors=SURFACE, zorder=5, linewidths=0.8, label="goal (goal-image pose)")
    if g.get("target") is not None:
        ax.scatter((g["target"][:2] - start[:2]) @ u, g["target"][2], s=25 if compact else 45, marker="x", color=INK2,
                   zorder=5, linewidths=1.2, label="env target")


def side_limits(g, pad=0.04):
    start, goal = g["path"][0], g["goal"]
    u = (goal[:2] - start[:2]) / max(1e-9, np.linalg.norm(goal[:2] - start[:2]))
    xs = list((g["path"][:, :2] - start[:2]) @ u) + [(goal[:2] - start[:2]) @ u]
    zs = list(g["path"][:, 2]) + [goal[2]]
    if g.get("target") is not None:
        xs.append((g["target"][:2] - start[:2]) @ u)
        zs.append(g["target"][2])
    if g["obstacle"] is not None and g["top_z"] is not None:
        xs += list((g["obstacle"] - start[:2]) @ u)
        zs += [g["top_z"], min(zs) - 0.04]
    return (min(xs) - pad, max(xs) + pad), (min(zs) - pad, max(zs) + pad)


def episode_video(plt, e, g, path):
    """One frame per state (start, then after each executed step): goal and current images of both cameras,
    top-down and side view of the path so far, distance-to-goal curve."""
    import imageio.v2 as imageio
    t = e["dir"]
    cams = [read_frames(p) for p in sorted(t.glob("episode_*_img_*_raw.mp4"))]
    cams = [c for c in cams if len(c) >= 2]
    if not cams:
        return False
    # Raw camera videos hold [goal, start, after step 1, ..., after step T], the same order as `poses`.
    goals = [c[0] for c in cams]
    T = min(len(c) for c in cams) - 1  # number of states shown (start + executed steps)
    dist = 100 * np.array(e["distance_by_step"]) if e.get("distance_by_step") else None
    start_d = 100 * e["start_distance_m"] if e.get("start_distance_m") else (dist[0] if dist is not None else 0)
    thr = 100 * e.get("success_distance_m", 0.05)
    frames = []
    for k in range(T):
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.8], hspace=0.35, wspace=0.18)
        for j, c in enumerate(cams[:2]):
            for i, (name, img) in enumerate(((f"goal · camera {j + 1}", goals[j]), (f"now · camera {j + 1}", c[k + 1]))):
                ax = fig.add_subplot(gs[i, j])
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                ax.set_title(name, fontsize=9, loc="center", fontweight="normal")
        if g is not None:
            upto = min(k, len(g["path"]) - 1)
            ax = fig.add_subplot(gs[0:2, 2])
            draw_topdown(ax, g, upto=upto, compact=True)
            lo, hi = frame_limits([g])
            ax.set_xlim(lo[0], hi[0])
            ax.set_ylim(lo[1], hi[1])
            ax.set_title("top-down", fontsize=9, loc="center")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, fontsize=7.5)
            ax = fig.add_subplot(gs[0:2, 3])
            draw_side(ax, g, upto=upto, compact=True)
            (x0, x1), (z0, z1) = side_limits(g)
            ax.set_xlim(x0, x1)
            ax.set_ylim(z0, z1)
            ax.set_xlabel("along start → goal (m)", fontsize=8)
            ax.set_ylabel("height (m)", fontsize=8)
            ax.set_title("side view", fontsize=9, loc="center")
        ax = fig.add_subplot(gs[2, :])
        if dist is not None:
            x = np.arange(0, len(dist) + 1)
            y = np.concatenate([[start_d], dist])
            ax.plot(x, y, color=GRID, linewidth=1.5)
            ax.plot(x[: k + 1], y[: k + 1], color=BLUE)
            ax.scatter([k], [y[min(k, len(y) - 1)]], s=30, color=BLUE, zorder=4)
            ax.axhline(thr, color=AQUA, linestyle="--", linewidth=1)
            ax.text(len(dist), thr, f" {thr:.0f} cm", color=AQUA, fontsize=8, va="center")
            ax.set_xlim(0, len(dist))
            ax.set_ylim(0, max(y) * 1.1)
        ax.set_xlabel("executed step")
        ax.set_ylabel(f"distance {REFERENCE.get(e.get('distance_reference'), '')} (cm)", fontsize=8)
        cur = (start_d if k == 0 else dist[min(k, len(dist)) - 1]) if dist is not None else math.nan
        mark = "✓ success" if e["success"] else "✗ fail"
        fig.suptitle(f"N = {e['samples']} · episode {e['seed']} · step {k}/{T - 1} · distance {cur:.1f} cm"
                     f"      final: {mark}, {100 * e['final_distance_m']:.1f} cm", x=0.01, ha="left", fontsize=12,
                     fontweight="bold", color=INK)
        # Planning time: the decision that produced this state, and the total so far (the video is not real time).
        lat = [d["latency_s"] for d in e.get("decisions", [])]
        this = f"planning for this step {lat[k - 1]:.1f} s" if 0 < k <= len(lat) else "start state"
        fig.text(0.01, 0.945, f"{this} · planning so far {sum(lat[:k]):.0f} s of {sum(lat):.0f} s · "
                 f"one frame per executed step (the sim only renders after each step)", fontsize=9, color=INK2)
        fig.subplots_adjust(left=0.04, right=0.98, top=0.9, bottom=0.08)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    fps = 3
    frames = [frames[0]] * fps + frames + [frames[-1]] * (2 * fps)  # hold the start 1 s and the end 2 s
    imageio.mimwrite(path, frames, fps=fps, macro_block_size=8)
    return True


# ---------------------------------------------------------------- tables

def fmt(v, spec):
    return "–" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, spec)


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def compare_note(runs, cmp_runs):
    if not cmp_runs:
        return []
    base = runs[0]["config"]
    diffs = {k: [r["config"].get(k) for r in cmp_runs] for k in COMPARE_KEYS
             if any(r["config"].get(k) != base.get(k) for r in cmp_runs)}
    names = ", ".join(r["label"] for r in cmp_runs)
    if not diffs:
        return [f"Compared with {names} (◆ in the plots, last columns in the tables): same task, scene, seeds and "
                "planner settings; only the model and N differ.", ""]
    text = "; ".join(f"`{k}` {base.get(k)} here vs {v}" for k, v in diffs.items())
    return [f"Compared with {names} (◆ in the plots). **Settings differ:** {text}.", ""]


def markdown(runs, rows, episodes, figures, root, cmp_runs=(), cmp_rows=()):
    c = runs[0]["config"]
    thr = 100 * c.get("success_distance", 0.05)
    series = success_series(rows[0])
    head = f"# {c['task']} · {c['model_size']} · candidates ladder"
    if cmp_runs:
        head += " · vs " + ", ".join(r["label"] for r in cmp_runs)
    lines = [head, "", setting_text(runs[0]), "", *compare_note(runs, cmp_runs),
             f"Success = final end-effector distance {ref_text(rows)} ≤ {thr:.0f} cm (provisional). The goal-image "
             "pose is where the arm was when the goal image was rendered (what the planner is asked to match); the "
             "env target is the env's sampled point used by its reward. They differ when the env's expert stopped "
             f"short while rendering the goal: mean gap {fmt(100 * rows[0]['goal_pose_to_target_mean_m'], '.1f')} cm "
             "in this run. "
             f"The {' and '.join(f'{cm:.0f}' for cm, *_ in series[1:])} cm columns are looser secondary "
             "thresholds (near misses count). "
             "Time = wall-clock planning time per executed step (decision); predictor = synchronized time inside "
             "predictor forwards.", "",
             "| model | N | batches × size | elites | episodes | "
             + " | ".join(f"success ≤ {cm:.0f} cm" for cm, *_ in series)
             + " | final distance cm, mean ± SD | final distance cm, median [IQR] | to env target cm (median) | "
             "planning s/decision | predictor s/decision | predictor ms/call | episode min |",
             "|---|---|---|---|---|" + "---|" * len(series) + "---|---|---|---|---|---|---|"]
    for r in [*rows, *cmp_rows]:
        failed = f" (+{r['failed']} failed)" if r["failed"] else ""
        succ = " | ".join(f"{r[k]}/{r['episodes']} ({fmt(100 * r[rate], '.0f')}%, CI {fmt(100 * r[lo], '.0f')}–"
                          f"{fmt(100 * r[hi], '.0f')})" for _, k, rate, lo, hi in series)
        model = f"**{r['model']}**" if r["compare"] else r["model"]
        lines.append(
            f"| {model} | {r['samples']} | {r['batches']} × {r['batch_size']} | {r['elites']} | {r['episodes']}{failed} | "
            f"{succ} | "
            f"{fmt(100 * r['final_distance_mean_m'], '.1f')} ± {fmt(100 * r['final_distance_sd_m'], '.1f')} | "
            f"{fmt(100 * r['final_distance_median_m'], '.1f')} "
            f"[{fmt(100 * r['final_distance_q1_m'], '.1f')}–{fmt(100 * r['final_distance_q3_m'], '.1f')}] | "
            f"{fmt(100 * r['final_distance_target_median_m'], '.1f')} | "
            f"{fmt(r['planning_s_per_decision'], '.1f')} | {fmt(r['predictor_s_per_decision'], '.1f')} | "
            f"{fmt(r['predictor_ms_per_call'], '.1f')} | {fmt(r['episode_wall_min'], '.1f')} |")
    lines += [""]
    for name, caption in figures:
        lines += [f"## {caption}", "", f"![{caption}]({name})", ""]
    seeds = sorted({e["seed"] for e in episodes})
    if seeds:
        lines += ["## Final distance per episode (cm; ✓ = success)", "",
                  "| episode | " + " | ".join(r["label"] if r["compare"] else f"N={r['samples']}"
                                              for r in [*rows, *cmp_rows]) + " |",
                  "|---|" + "---|" * (len(rows) + len(cmp_rows))]
        for s in seeds:
            cells = []
            for run in [*runs, *cmp_runs]:
                e = next((x for x in run["episodes"] if x["seed"] == s), None)
                if e is None:
                    cells.append("failed" if s in run["failed"] else "–")
                    continue
                link = f"episodes/{run['name']}_seed{s:05d}/"
                cells.append(f"[{'✓' if e['success'] else '✗'} {100 * e['final_distance_m']:.1f}]({link})")
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
        lines += ["", "Each link holds `trajectory.png` (top-down + side view with the obstacle) and `video.mp4`.", ""]
    (root / "README.md").write_text("\n".join(lines))


def report(root, videos=True, force=False, quiet=False, compare=None):
    """compare=None reuses the comparison runs saved in report/report.json; a list (possibly empty) replaces them."""
    root = Path(root)
    runs = [r for r in load_runs(root) if r["episodes"]]
    if not runs:
        raise SystemExit(f"no completed episodes under {root}")
    out = root / "report"
    out.mkdir(exist_ok=True)
    saved = out / "report.json"
    if compare is None:
        compare = [root / p for p in json.loads(saved.read_text()).get("compare", [])] if saved.exists() else []
    else:
        compare = [Path(p).resolve() for p in compare]
        saved.write_text(json.dumps({"compare": [os.path.relpath(p, root) for p in compare]}, indent=2) + "\n")
    for r in runs:
        r["label"] = f"{r['config']['model_size'].upper()}, N = {r['config']['samples']}"
    cmp_runs = []
    for d in compare:
        for r in load_runs(d):
            if r["episodes"]:
                r["compare"] = True
                r["name"] = f"{r['config']['model_size']}_{r['name']}"
                r["label"] = f"{r['config']['model_size'].upper()}, N = {r['config']['samples']}"
                cmp_runs.append(r)
    plt = plt_setup()
    rows = [run_row(r) for r in runs]
    cmp_rows = [run_row(r) for r in cmp_runs]
    for r in [*runs, *cmp_runs]:
        for e in r["episodes"]:
            e["run"] = r["name"]
    episodes = [e for r in [*runs, *cmp_runs] for e in r["episodes"]]
    write_csv(out / "summary.csv", [*rows, *cmp_rows])
    write_csv(out / "episodes.csv", [{k: v for k, v in e.items() if not isinstance(v, (list, dict, Path))}
                                     for e in episodes])
    figures = []
    if len(rows) > 1:
        plot_time(plt, runs, rows, out, cmp_rows)
        success_panels(plt, runs, rows, "samples", "candidate action sequences per CEM iteration (N)",
                       "Does more candidates raise success?", out / "success_vs_samples.png", log_x=True,
                       cmp_runs=cmp_runs, cmp_rows=cmp_rows)
        success_panels(plt, runs, rows, "planning_s_per_decision", "planning seconds per decision",
                       "Success vs planning time", out / "success_vs_time.png", log_x=False,
                       cmp_runs=cmp_runs, cmp_rows=cmp_rows)
        figures += [("time_vs_samples.png", "Planning time vs candidates"),
                    ("success_vs_samples.png", "Success vs candidates"),
                    ("success_vs_time.png", "Success vs planning time")]
    plot_distance_steps(plt, runs, out, cmp_runs)
    figures.append(("distance_vs_step.png", "Distance to goal during the episode"))
    if plot_trajectory_grid(plt, [*runs, *cmp_runs], out):
        figures.append(("trajectories.png", "End-effector paths with the obstacle"))
    for run in [*runs, *cmp_runs]:
        for e in run["episodes"]:
            d = out / "episodes" / f"{run['name']}_seed{e['seed']:05d}"
            d.mkdir(parents=True, exist_ok=True)
            e["success_distance_m"] = run["config"].get("success_distance", 0.05)
            # Episode outputs never change once the trial is done: render only new ones (refresh is cheap).
            g = episode_geometry(e)
            if g is not None and (force or not (d / "trajectory.png").exists()):
                plot_episode(plt, e, g, d / "trajectory.png")
            if videos and (force or not (d / "video.mp4").exists()):
                episode_video(plt, e, g, d / "video.mp4")
    markdown(runs, rows, episodes, figures, out, cmp_runs, cmp_rows)
    if not quiet:
        print(f"report: {out / 'README.md'}")
    return out
