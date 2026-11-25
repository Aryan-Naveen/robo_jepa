# Frozen RoboJEPA experiments

A single entry point for RoboCasa planning trials, with the local ObstacleReach adjustments built in. Run from the repository root. The upstream training/model/simulator code and its history are retained; the portable experiment layer is entirely in `experiments/`.

## Setup

Python 3.11, Linux, an NVIDIA CUDA GPU, RoboCasa assets, a frozen predictor checkpoint, and the frozen V-JEPA 2.1 encoder are required. The local development machine uses an RTX 5090 (32 GiB), PyTorch 2.7.1 and CUDA 12.8. For that setup:

```bash
conda env create -f experiments/environment-cu128.yml
conda activate robo_jepa
export PYTHONNOUSERSITE=1
python -m robocasa.scripts.setup_macros
python -m robosuite.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets
```

The upstream `environment.yml` is the CUDA 12.6 alternative. The environment includes upstream training dependencies too; these are retained to avoid breaking imports in the existing policy client. Cosmos decoder weights and tokenizer are not needed for this interface. Setup needs internet access and about 23 GB for kitchen assets, in addition to packages and model weights.

Download a model (300M example):

```bash
mkdir -p models/wm-scaling-300m-norms
cp configs/released/wm-300m/config.json models/wm-scaling-300m-norms/config.json
curl -fL --retry 3 -o models/wm-scaling-300m-norms/cooldown_300m_300k.pth.tar \
  https://dl.fbaipublicfiles.com/robojepa/wm/wm-300m/cooldown_300m_300k.pth.tar
curl -fL --retry 3 -o models/vjepa2_1_vitG_384.pt \
  https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt
ln -s ../vjepa2_1_vitG_384.pt models/wm-scaling-300m-norms/encoder.pth.tar
```

Other model URLs/checkpoint steps are in the upstream root README. Models default to these final steps: 22M/150k, 50M/120k, 100M/150k, 300M/300k, 1B/300k, 2B/450k, 4B/450k, 8B/300k. Override with `--checkpoint-step`. Alternatively use `--checkpoint-dir DIR` containing `config.json`, `world_model.pth.tar`, and `encoder.pth.tar`; in that mode `--model-size` is a reporting label and you must match it to the supplied weights.

## Run

```bash
python -m experiments run \
  --task obstacle_reach --model-size 300m --trials 3 \
  --models-dir models --output outputs/obstacle-300m
```

The main arguments are `--task {reach,object_reach,obstacle_reach,push}`, `--model-size {22m,50m,100m,300m,1b,2b,4b,8b}`, and `--trials N`. All parameters are available via `python -m experiments run --help`.

Useful variations:

```bash
# Cheap pipeline smoke test, not a task-performance measurement
python -m experiments run --task obstacle_reach --model-size 22m --trials 1 \
  --samples 4 --batch-size 4 --planning-steps 1 --horizon 1 \
  --cem-iterations 1 --output outputs/smoke

# Change task, model, scene, seeds, and CEM budget
python -m experiments run --task reach --model-size 50m --trials 10 \
  --scene 0 --seed 10 --samples 64 --output outputs/reach-50m

# Smaller batches for a larger model (total candidate budget stays 1024)
python -m experiments run --model-size 2b --batch-size 8 --trials 3 \
  --output outputs/obstacle-2b

# Inspect all resolved options and exact subprocess commands, without GPU/model loading
python -m experiments run --output outputs/example --dry-run

# Repeat the exact original command with --resume to skip completed trials
python -m experiments run --task obstacle_reach --model-size 300m --trials 3 \
  --models-dir models --output outputs/obstacle-300m --resume

python -m experiments summarize outputs/obstacle-300m
```

A trial is one task seed and one episode, with task and planner RNG seeds `seed .. seed+trials-1`. Scene remains fixed unless changed with `--scene`. The planner RNG is reseeded at the first CEM call, independently of model-loading RNG consumption. A fresh simulator and planner process per trial isolates state and failures, but reloads weights. Encoder and predictor are explicitly frozen and put in eval mode. This interface supports single-GPU eager execution; larger models may need more GPU memory. Separate simultaneous runs need distinct `--port`, `--master-port`, and output directories.

The launcher waits for the simulator's listening message, bounds startup/trial duration, checks exit codes, and terminates its process groups on errors or interruption. A failed trial stops the run; `--resume` retries it. Existing output is never silently overwritten. Resume requires identical arguments and unchanged checkpoint source metadata. The source commit, dirty status, model config, checkpoint locations/sizes/mtimes, runtime versions, and commands are saved. Checkpoint identity is based on metadata, not a cryptographic weight hash; resuming across code revisions should be avoided.

## Defaults and local adjustments

| Task | Execution steps | Horizon | Scene | Task settings |
|---|---:|---:|---:|---|
| Reach | 5 | 1 | 0 | max distance 30 cm |
| ObjectReach | 5 | 1 | 0 | banana, max distance 15 cm |
| ObstacleReach | **18** | 5 | 0 | **25–50 cm**, **3 cm clearance**, CoffeeMachine, target kitchens |
| Push | 18 | 5 | 4 | flourbag001, 60-degree forward cone |

All tasks use state-delta embodiment ID **3**. This preserves the outer sweep's correction for Reach/ObjectReach, whose upstream examples default to action-key ID 2. The three bold ObstacleReach settings replace the old example's 5 steps, 40–80 cm band, and implicit clearance. They were previously applied through `sed` in the local `band25` sweep. They are experimental reproduction adjustments, not proof of matching every paper condition.

Shared defaults: CEM budget 1024, batch 16, first/later iterations 15/10, elite count min(20, max(2, budget/4)), action norm 0.3, rollout cache enabled, translation only, two equally weighted cameras, one executed action per planning decision. Use `--samples`, `--batch-size`, `--horizon`, `--planning-steps`, `--reach-min-distance`, `--reach-max-distance`, and `--obstacle-clearance` to vary these settings.

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

Unit tests cover task configuration, invalid budgets, FLOP counting/output preservation, token/memory aggregation, missing success flags, and success-only step counts. Use the smoke command above to validate your installed GPU/simulator environment. Full paper-scale results need a larger evaluation across scenes/seeds and have not been established by these smoke tests.

Validation on the local RTX 5090: all seven unit tests passed; a one-step ObstacleReach trial with FLOP counting, a two-step cached ObstacleReach trial, and two two-step Reach trials without FLOP counting completed successfully. Resume skipped both completed Reach trials. These reduced-budget runs validate the pipeline, not policy quality.
