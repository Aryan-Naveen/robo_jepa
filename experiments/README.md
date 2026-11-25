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

Shared defaults: CEM budget 1024, batch 16, first/later iterations 15/10, elite count min(20, max(2, budget/4)), action norm 0.3, rollout cache enabled, translation only, two equally weighted cameras, one executed action per planning decision. Use `--samples`, `--batch-size`, `--horizon`, `--planning-steps`, `--reach-min-distance`, `--reach-max-distance`, and `--obstacle-clearance` to vary these settings.

## Execution and resume behavior

A trial is one task seed and one episode, with task and planner RNG seeds `seed .. seed+trials-1`. Scene remains fixed unless changed with `--scene`. The planner RNG is reseeded at the first CEM call, independently of model-loading RNG consumption. A fresh simulator and planner process per trial isolates state and failures, but reloads weights. Encoder and predictor are explicitly frozen and put in eval mode. This interface supports single-GPU eager execution; larger models may need more GPU memory. Separate simultaneous runs need distinct `--port`, `--master-port`, and output directories.

The launcher waits for the simulator's listening message, bounds startup/trial duration, checks exit codes, and terminates its process groups on errors or interruption. A failed trial stops the run; `--resume` retries it. Existing output is never silently overwritten. Resume requires identical arguments and unchanged checkpoint source metadata. The source commit, dirty status, model config, checkpoint locations/sizes/mtimes, runtime versions, and commands are saved. Checkpoint identity is based on metadata, not a cryptographic weight hash; resuming across code revisions should be avoided.

## Outputs and metric definitions

Every run contains `summary.json`, `trials.csv`, and `trial_<seed>/` directories with `metrics.json`, raw `efficiency.json`, simulator/planner logs, hardware metadata, NPZ trajectories and videos. `metrics.json` includes predictor totals for each planning decision. The summary pools module calls over completed trials.

| Metric | Definition |
|---|---|
| Success rate (SR) | Fraction of completed episodes with any simulator `success=True`; failures to execute are listed separately and excluded from SR. Check completed/requested counts before comparing runs. |
| Task progress | Upstream normalized reward: mean per-action reward for Reach/ObjectReach, final reward for ObstacleReach/Push. Final, mean, and per-step rewards are also saved. |
| Steps to success | 1-based index of first successful executed action, which equals planning decision index with this runner's execution prefix/substeps both fixed to 1. Null for unsuccessful episodes; mean includes successes only. Episodes still run their fixed budget after first success. |
| Output TPS | Sum of output visual latent tokens / sum of measured module forward times. One token is one embedding vector, excluding feature channels; predictor context output positions and all sampled candidates count. Not text tokens or successful robot actions. |
| Predictor latency | Synchronized wall time around the predictor forward, excluding state preprocessing/transfers outside that call. Includes CPU dispatch and profiling overhead. Total across all sampled actions, batch-call mean, and total time / sample-forwards are saved. The last is amortized latency, **not** independently measured batch-1 latency. |
| Encoder latency | Same measurement around encoder forward; includes goal, initial observation, and subsequent observation encodings. Image preprocessing outside the encoder is excluded. |
| Planning latency | Entire CEM call, including sampling/scoring and module instrumentation; does not include simulator execution or observation encoding. |
| Average memory | Mean PyTorch allocated GPU bytes sampled immediately before/after module calls. A boundary-sampled average, not a continuous time-weighted average. |
| Peak memory | Highest process allocated GPU bytes observed during that module's forwards. Includes resident weights and other tensors. Also reports peak increase above the pre-call baseline. These are **not** exclusive per-module memory totals and exclude simulator/driver/reserved allocations. |
| FLOPs/MACs | `torch.utils.flop_counter.FlopCounterMode` counts supported operations during actual forwards; MAC-equivalents = FLOPs/2. Includes supported attention/linear operations, not a complete hardware instruction count. |

FLOP counting is enabled by default and adds overhead. For latency comparisons use a separate `--no-count-flops` run; FLOPs/MACs then appear as null. Raw timing includes cold first calls; no hidden warmup is discarded. Both profiler mode and raw per-call records are saved so comparisons can use the same conditions. Single-GPU execution and synchronized measurement intentionally prioritize interpretable measurements over maximum asynchronous throughput.

## Previous experiment state

`baselines/obstacle_band25_300m.json` preserves the completed local 300M budget ladder: scene 0, seeds 0–2, one episode each, 18 steps, horizon 5, 25–50 cm band, 3 cm clearance. At N=1024 final rewards were 0.7701, 0.0189, 0.9811 (mean 0.5901). The new runner preserves the original first-CEM reseeding rule, but isolates trials in separate processes and changes instrumentation; these are reference results, not bit-for-bit expectations or evidence of full paper reproduction. Raw old videos/checkpoints/logs remain outside this fork. The original sweep's summary included old defaults and is deliberately not the new summary.

## Validation and development

```bash
python -m unittest discover -s experiments/tests -v
python -m compileall -q experiments
```

Unit tests cover task configuration, invalid budgets, FLOP counting/output preservation, token/memory aggregation, missing success flags, and success-only step counts. Use the smoke command in the [root README](../README.md#3-run-experiments) to validate your installed GPU/simulator environment. Full paper-scale results need a larger evaluation across scenes/seeds and have not been established by these smoke tests.

Validation on the local RTX 5090: all seven unit tests passed; a one-step ObstacleReach trial with FLOP counting, a two-step cached ObstacleReach trial, and two two-step Reach trials without FLOP counting completed successfully. Resume skipped both completed Reach trials. These reduced-budget runs validate the pipeline, not policy quality.
