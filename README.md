# RoboJEPA experiment interface

Run frozen RoboJEPA models on RoboCasa tasks and measure robot performance alongside encoder and predictor efficiency. This fork provides one command-line interface, reproducible trial settings, and JSON/CSV results, including the local ObstacleReach adjustments.

Built on **[facebookresearch/robo_jepa](https://github.com/facebookresearch/robo_jepa)**, the original RoboJEPA repository. See its [README](https://github.com/facebookresearch/robo_jepa#readme) for the paper, model catalog, training instructions, and upstream deployment tools. The original model and simulator code is retained here; this fork's experiment interface lives in [`experiments/`](experiments/).

## 1. Clone and install

```bash
git clone https://github.com/Aryan-Naveen/robo_jepa.git
cd robo_jepa
```

Run all commands below from this repository root.

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

## 2. Download checkpoints

The following example prepares the 300M predictor and its shared frozen encoder:

```bash
mkdir -p models/wm-scaling-300m-norms
cp configs/released/wm-300m/config.json models/wm-scaling-300m-norms/config.json
curl -fL --retry 3 -o models/wm-scaling-300m-norms/cooldown_300m_300k.pth.tar \
  https://dl.fbaipublicfiles.com/robojepa/wm/wm-300m/cooldown_300m_300k.pth.tar
curl -fL --retry 3 -o models/vjepa2_1_vitG_384.pt \
  https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitG_384.pt
ln -s ../vjepa2_1_vitG_384.pt models/wm-scaling-300m-norms/encoder.pth.tar
```

Other model URLs and checkpoint steps are listed in the [original RoboJEPA README](https://github.com/facebookresearch/robo_jepa#readme). Models default to these final steps: 22M/150k, 50M/120k, 100M/150k, 300M/300k, 1B/300k, 2B/450k, 4B/450k, 8B/300k. Override with `--checkpoint-step`. Alternatively use `--checkpoint-dir DIR` containing `config.json`, `world_model.pth.tar`, and `encoder.pth.tar`; in that mode `--model-size` is a reporting label and you must match it to the supplied weights.

## 3. Run experiments

```bash
python -m experiments run \
  --task obstacle_reach --model-size 300m --trials 3 \
  --models-dir models --output outputs/obstacle-300m
```

The main arguments are `--task {reach,object_reach,obstacle_reach,push}`, `--model-size {22m,50m,100m,300m,1b,2b,4b,8b}`, and `--trials N`. All parameters are available via `python -m experiments run --help`.

Useful variations:

```bash
# Cheap pipeline smoke test, not a task-performance measurement
python -m experiments run --task obstacle_reach --model-size 300m --trials 1 \
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

# Candidates-per-iteration ladder: quick check first, then the full run; then tables, plots and videos
python -m experiments run --preset smoke --model-size 100m --output outputs/ladder-smoke
python -m experiments run --preset full --model-size 100m --output outputs/ladder-100m
python -m experiments report outputs/ladder-100m      # -> outputs/ladder-100m/report/README.md
```

See [candidates ladder and report](experiments/README.md#candidates-ladder-and-report) (adding N to an existing ladder, cost per episode).

Each trial runs one episode with a distinct task seed; the scene stays fixed. Encoder and predictor weights remain frozen. Runs use a single GPU in eager mode. Download the corresponding checkpoints before switching model sizes.

A failed trial is recorded in `failure.json` and the run continues. Repeat the same command with `--resume` to retry it and skip completed trials. Resume requires identical settings and unchanged checkpoint files. For concurrent runs, choose distinct output directories, `--port`, and `--master-port` values.

## Configure a run

| Argument | Purpose | Default |
|---|---|---|
| `--task` | `reach`, `object_reach`, `obstacle_reach`, or `push` | `obstacle_reach` |
| `--model-size` | Frozen predictor size, from `22m` through `8b` | `300m` |
| `--trials` | Number of episodes, one per task seed | `3` |
| `--scene` / `--seed` | Fixed kitchen and first task/planner seed | Scene `0` (`4` for push), seed `0` |
| `--preset` | `smoke` (quick pipeline check) or `full` (candidates ladder); fills only unset options | none |
| `--samples` | Total CEM candidates per iteration; several values run a ladder | `1024` |
| `--batch-size` | Candidates evaluated together; must divide `--samples` | `16` |
| `--planning-steps` | Maximum executed planning decisions | `18` for obstacle/push, `5` otherwise |
| `--horizon` | Predicted rollout length | `5` for obstacle/push, `1` otherwise |
| `--models-dir` | Directory containing downloaded model folders | `models` |
| `--output` | Run directory for metrics, logs, and trajectories | Required |
| `--flops` | `once` per input shape (cheap), `all` (slow), or `off` | `once` |
| `--success-distance` | Reach tasks: success if the final distance to the goal is at most this (m) | `0.05` |
| `--success-reference` | Reach tasks: measure distance to the goal-image pose (`goal_pose`) or the env's sampled target (`target`) | `goal_pose` |
| `--no-fast-planner` | Keep the stock planner's per-batch `gc.collect()` and GPU->CPU state copy | fast planner on |

Additional flags control checkpoint selection, objects, obstacle distances, GPU, ports, timeouts, and CEM iteration counts. Inspect them with `python -m experiments run --help`.

**ObstacleReach defaults:** 18 execution steps, horizon 5, 25–50 cm start–goal distance, 3 cm obstacle clearance, and a CoffeeMachine obstacle in the target kitchen split. All tasks use state-delta embodiment ID 3. These preserve the local reproduction adjustments; full paper-scale reproduction remains unverified. See [task defaults and changes](experiments/README.md#defaults-and-local-adjustments).

## Read the results

```text
outputs/obstacle-300m/
├── config.json                   # resolved experiment settings
├── checkpoint_provenance.json    # model config and checkpoint metadata
├── revision.json                 # source revision and Python version
├── summary.json                  # aggregate task and efficiency metrics
├── trials.csv                    # one row per completed trial
└── trial_00000/                  # repeated for each seed
    ├── metrics.json              # task metrics and per-decision predictor totals
    ├── efficiency.json           # raw encoder/predictor measurements
    ├── hardware.json             # GPU, runtime, parameter counts
    ├── commands.json             # exact simulator and planner commands
    ├── server.log / planner.log
    └── episode_*.npz / *.mp4     # trajectory and videos
```

Robot metrics include **success rate (SR)**, **final distance to the goal**, **task progress**, and **planning steps to first success**. Encoder and predictor metrics include **output tokens/second**, **average/peak GPU allocation**, **FLOPs/MACs**, and **forward latency** across batches and amortized per sample.

Success rate excludes trials that failed to execute, so check completed/requested trial counts. Tokens are visual latent embeddings; per-sample latency is amortized batch latency. Average memory is sampled at call boundaries, and peak memory includes other tensors resident in the planner process. FLOPs are counted once per input shape and those calls are excluded from latency statistics. See [precise metric definitions](experiments/README.md#outputs-and-metric-definitions) before comparing results.

## Reference and validation

- [Experiment reference](experiments/README.md): task defaults, metric definitions, execution behavior, and previous results.
- [Historical ObstacleReach baseline](experiments/baselines/obstacle_band25_300m.json): the local 300M sample-budget ladder.
- [Original repository](https://github.com/facebookresearch/robo_jepa): upstream research, checkpoints, and training documentation.
- [License](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).

```bash
python -m unittest discover -s experiments/tests -v
python -m compileall -q experiments
```

The interface passed seven unit tests, repository static checks, and four reduced-budget GPU smoke-test episodes on an RTX 5090, including cached rollouts and resume behavior. These checks validate the experiment pipeline, not paper-level task performance.
