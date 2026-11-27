# Experiment reference

For installation, checkpoint downloads, and command-line examples, start with the **[root README](../README.md)**. This reference documents the experiment defaults, metric definitions, and historical results. The model and simulator implementation comes from [facebookresearch/robo_jepa](https://github.com/facebookresearch/robo_jepa).

## Defaults and local adjustments

| Task | Execution steps | Horizon | Scene | Task settings |
|---|---:|---:|---:|---|
| Reach | 5 | 1 | 0 | max distance 30 cm |
| ObjectReach | 5 | 1 | 0 | banana, max distance 15 cm |
| ObstacleReach | **18** | 5 | 0 | **25–50 cm**, **3 cm clearance**, CoffeeMachine, target kitchens |
| Push | 18 | 5 | 4 | flourbag001, 60-degree forward cone |

All tasks use state-delta embodiment ID **3**. This preserves the outer sweep's correction for Reach/ObjectReach, whose upstream examples default to action-key ID 2. The three bold ObstacleReach settings replace the old example's 5 steps, 40–80 cm band, and implicit clearance. They were previously applied through `sed` in the local `band25` sweep. They are experimental reproduction adjustments, not proof of matching every paper condition.

Shared defaults: CEM budget 1024, batch 16, first/later iterations 15/10, elites 2% of the candidates (20 at N=1024), at least 4 and at most N/2, action norm 0.3, rollout cache enabled, translation only, two equally weighted cameras, one executed action per planning decision. Use `--samples`, `--batch-size`, `--horizon`, `--planning-steps`, `--reach-min-distance`, `--reach-max-distance`, and `--obstacle-clearance` to vary these settings.

## Candidates ladder and report

The main question: does proposing more candidate action sequences per CEM iteration (N, default 1024 = 64 batches
of 16) raise success, and how much planning time does it cost? A ladder runs several N with everything else fixed and
the same episodes (task and planner seeds) for every N:

```bash
# 1. quick end-to-end check of runs, plots and videos (100M: N=16/64/256, 1 episode, 6 steps, CEM 3+2; ~15 min)
python -m experiments run --preset smoke --model-size 100m --output results/ladder_smoke_100m
python -m experiments report results/ladder_smoke_100m

# 2. the real ladder (N=16..1024, 10 episodes, 18 steps, CEM 15+10)
python -m experiments run --preset full --model-size 100m --output results/ladder_100m
python -m experiments report results/ladder_100m
```

Presets only fill options left unset, so `--preset full --trials 3 --samples 16 64 1024` works. Fairness rules:
N changes the number of sequential batches of 16 (same GPU shape per predictor call; N < 16 uses one smaller batch);
iterations, horizon and seeds stay fixed; elites are a fixed 2% fraction with a floor of 4 so that small N still
selects. Seeds run in the outer loop, so a partial ladder covers every N. A failed trial is recorded in
`failure.json` and the ladder continues; `--resume` retries it.

To add candidate counts to an existing ladder later, run again into the same directory with only the new N and the
same other options, e.g. `run --preset full --model-size 100m --samples 256 512 --output results/ladder_100m`. Each N
lives in its own `n<NNNN>/` subdirectory and the new counts are merged into `ladder.json`, so the report covers all of
them.

Cost: planning time is linear in N. Measured on one RTX 5090 (100M, ObstacleReach, 18 steps, horizon 5, CEM 15+10,
fast planner), one predictor call at batch 16 takes ~57 ms and an episode takes 3.6 / 5.2 / 8.5 / 14.8 min at
N = 16 / 32 / 64 / 128. Extrapolated: ~30 min at N=256, ~1 h at 512, ~1.6 h at 1024 per episode, so the full
16..1024 preset with 10 episodes is about 1.5 GPU-days at 100M. Larger models cost more per call.

`run` refreshes the report after every trial (`--no-report` turns this off), and `report` can be run at any time,
also during a run: it reads only finished episodes and renders each episode's trajectory and video once
(`--force` re-renders). `report` writes `<dir>/report/README.md` with the summary table (success at `--success-distance` and at looser 10 and 15 cm thresholds, each with a 95% Wilson interval;
final distance mean ± SD and median/IQR, planning and predictor seconds per decision, ms per predictor call), the plots `time_vs_samples.png`,
`success_vs_samples.png`, `success_vs_time.png`, `distance_vs_step.png`, `trajectories.png` (top-down paths with the
obstacle, episodes x N), and per episode `episodes/<run>_seed<k>/trajectory.png` (top-down + side view) and
`video.mp4` (goal and current camera images, path so far, distance curve). The obstacle footprint and goal come
from the env's reset observation, saved per trial in `scene.json`.

`report --compare <run or ladder> ...` adds other runs, typically another model size on the same seeds, as extra
rows in the tables, ◆ markers in the success, distance and time plots, and extra columns in the trajectory grid. The
report states which settings differ, if any. The list is saved in `report/report.json`, so later refreshes (including
the automatic one during `run`) keep it; `--compare` with no value clears it. Example:
`python -m experiments report results/ladder_100m --compare results/ladder_300m/n0064`.

Planner overhead fixes (`--fast-planner`, on by default; `experiments/fast_planner.py`): the rollout state stays on
the GPU instead of a ~64 MB GPU->CPU copy per step, and the per-batch `gc.collect()` (~0.16 s) is skipped. Numerics
are unchanged; both overheads grow with N and would otherwise hide predictor speed-ups.

## Execution and resume behavior

A trial is one task seed and one episode, with task and planner RNG seeds `seed .. seed+trials-1`. Scene remains fixed unless changed with `--scene`. The planner RNG is reseeded at the first CEM call, independently of model-loading RNG consumption. A fresh simulator and planner process per trial isolates state and failures, but reloads weights. Encoder and predictor are explicitly frozen and put in eval mode. This interface supports single-GPU eager execution; larger models may need more GPU memory. Separate simultaneous runs need distinct `--port`, `--master-port`, and output directories.

The launcher waits for the simulator's listening message, bounds startup/trial duration, checks exit codes, and terminates its process groups on errors or interruption. A failed trial is recorded and the run continues; `--resume` retries it. Existing output is never silently overwritten. Resume requires identical arguments and unchanged checkpoint source metadata. The source commit, dirty status, model config, checkpoint locations/sizes/mtimes, runtime versions, and commands are saved. Checkpoint identity is based on metadata, not a cryptographic weight hash; resuming across code revisions should be avoided.

## Outputs and metric definitions

Every run contains `summary.json`, `trials.csv`, and `trial_<seed>/` directories with `metrics.json`, raw `efficiency.json`, simulator/planner logs, hardware metadata, NPZ trajectories and videos. `metrics.json` includes predictor totals for each planning decision. The summary pools module calls over completed trials.

| Metric | Definition |
|---|---|
| Success rate (SR) | Reach / ObjectReach / ObstacleReach: fraction of completed episodes whose **final** end-effector distance to the goal is ≤ `--success-distance` (default 5 cm, provisional). The goal is the end-effector position shown in the goal image (`--success-reference goal_pose`, default); `--success-reference target` uses the env's sampled target instead (see [goal image vs env target](#goal-image-vs-env-target)). The simulator's own `success` flag is that of the host task these reach tasks reuse as a backdrop (CloseStandMixerHead), so it is always False; it is kept as `env_success_any_step`. Push: any simulator `success=True`. Failed runs are listed separately and excluded from SR. |
| Task progress | Upstream normalized reward: mean per-action reward for Reach/ObjectReach, final reward for ObstacleReach/Push. Final, mean, and per-step rewards are also saved. |
| Final distance | End-effector distance (simulator world coordinates, metres) to the success reference after the last step; also saved per step (`distance_by_step`). Both variants are saved: `final_distance_goal_pose_m` and `final_distance_target_m` (the env's `reach_distance_to_goal`), plus the gap between the two goals, `goal_pose_to_target_m`. |
| Steps to success | 1-based index of the first executed action that meets the success condition, which equals planning decision index with this runner's execution prefix/substeps both fixed to 1. Null for unsuccessful episodes; mean includes successes only. Episodes still run their fixed budget after first success. |
| Output TPS | Sum of output visual latent tokens / sum of measured module forward times. One token is one embedding vector, excluding feature channels; predictor context output positions and all sampled candidates count. Not text tokens or successful robot actions. |
| Predictor latency | Synchronized wall time around the predictor forward, excluding state preprocessing/transfers outside that call. Includes CPU dispatch and profiling overhead. Total across all sampled actions, batch-call mean, and total time / sample-forwards are saved. The last is amortized latency, **not** independently measured batch-1 latency. |
| Encoder latency | Same measurement around encoder forward; includes goal, initial observation, and subsequent observation encodings. Image preprocessing outside the encoder is excluded. |
| Planning latency | Entire CEM call, including sampling/scoring and module instrumentation; does not include simulator execution or observation encoding. |
| Average memory | Mean PyTorch allocated GPU bytes sampled immediately before/after module calls. A boundary-sampled average, not a continuous time-weighted average. |
| Peak memory | Highest process allocated GPU bytes observed during that module's forwards. Includes resident weights and other tensors. Also reports peak increase above the pre-call baseline. These are **not** exclusive per-module memory totals and exclude simulator/driver/reserved allocations. |
| FLOPs/MACs | `torch.utils.flop_counter.FlopCounterMode` counts supported operations; MAC-equivalents = FLOPs/2. Includes supported attention/linear operations, not a complete hardware instruction count. |

`--flops once` (default) counts the first forward of each input shape and reuses that count for later forwards of
the same shape; those few counted forwards are flagged (`flops_counted`) and left out of the latency statistics,
because FlopCounterMode intercepts every op in Python (it roughly doubled predictor call time when applied to every
call). `--flops all` counts every forward (slow); `--flops off` disables counting. Raw timing includes cold first calls; no hidden warmup is discarded. Both profiler mode and raw per-call records are saved so comparisons can use the same conditions. Single-GPU execution and synchronized measurement intentionally prioritize interpretable measurements over maximum asynchronous throughput.

## Previous experiment state

`baselines/obstacle_band25_300m.json` preserves the completed local 300M budget ladder: scene 0, seeds 0–2, one episode each, 18 steps, horizon 5, 25–50 cm band, 3 cm clearance. At N=1024 final rewards were 0.7701, 0.0189, 0.9811 (mean 0.5901). The new runner preserves the original first-CEM reseeding rule, but isolates trials in separate processes and changes instrumentation; these are reference results, not bit-for-bit expectations or evidence of full paper reproduction. Raw old videos/checkpoints/logs remain outside this fork. The original sweep's summary included old defaults and is deliberately not the new summary.

## Goal image vs env target

The reach tasks build the goal live: the env samples a target point, drives the arm toward it with a scripted
controller, and renders the goal image wherever the arm stops. The planner only sees that image, but the env's reward
and `reach_distance_to_goal` measure distance to the sampled target. When the controller stops short (on ObstacleReach
scene 0 this happens for 5 of seeds 0–9, by 12–20 cm, always vertically behind the obstacle), even perfect imitation of
the goal image scores as a miss against the target. The runner therefore scores against the goal-image pose by default
and reports the gap per episode (`goal_pose_to_target_m`, from `scene.json`). The gap also changes difficulty: goals
where the arm stopped high are easier than goals at counter level. Compare methods only on the same seeds.

## Validation and development

```bash
python -m unittest discover -s experiments/tests -v
python -m compileall -q experiments
```

Unit tests cover task configuration, invalid budgets, FLOP counting/output preservation (including count-once-per-shape), token/memory aggregation, missing success flags, success-only step counts, reach success from final distance (both references), and ladder/elite resolution. Use the smoke command in the [root README](../README.md#3-run-experiments) to validate your installed GPU/simulator environment. Full paper-scale results need a larger evaluation across scenes/seeds and have not been established by these smoke tests.

Validation on the local RTX 5090: the original seven unit tests passed; a one-step ObstacleReach trial with FLOP counting, a two-step cached ObstacleReach trial, and two two-step Reach trials without FLOP counting completed successfully. Resume skipped both completed Reach trials. These reduced-budget runs validate the pipeline, not policy quality. Later, all 10 unit tests passed, the smoke ladder ran end to end, and a 100M ObstacleReach ladder (N = 16/32/64/128, seeds 0–9) completed with no failed trials.
