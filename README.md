# MARL racing

Reinforcement learning experiments for racing.

Physics baseline for the F0–F2 geometry correction: `origin/multiNode` at
`7d65c76fadff9081ce59c4a42c7400b892adc92e`. Default dynamics remain
`combined_slip_st` v2 / MF6.1, RK4 with 0.001 s maximum substeps and a 0.05 s
simulation step (standalone simulator default: 0.01 s). Its chassis is 0.58 ×
0.31 m, with axles at +0.15875 / −0.17145 m; the legacy preset keeps its
0.32 × 0.225 m footprint and v1 dynamics. Run provenance records the independent
`vehicle_geometry_collision` behavior version (2); dynamics/configs are unchanged.

## Local PPO and MAPPO

Run these commands from the repository in the project’s Python environment.
Add `--set execution.params.device=cpu` for CPU learning, or
`--set execution.params.device=cuda` for GPU learning. Add `--resolve-config`
to inspect a scenario and its overrides before training.

```bash
# Solo PPO pretraining
python run.py --scenario scenarios/ppo_lap_completion_pretrain.yaml \
  --output-dir outputs/pretrain

# PPO full fine-tuning on the transfer map set
python run.py --scenario scenarios/ppo_lap_completion_transfer.yaml \
  --checkpoint outputs/pretrain/best.pt --output-dir outputs/transfer

# MAPPO from scratch
python run.py --scenario scenarios/mappo_2v2_completion_scratch.yaml \
  --output-dir outputs/mappo_scratch

# MAPPO transfer with per-agent LoRA
python run.py --scenario scenarios/mappo_2v2_completion_lora.yaml \
  --pretrained-actor outputs/pretrain/best.pt --output-dir outputs/mappo_lora
```

Use `ppo_lap_completion_transfer_3lap.yaml` for the three-lap transfer preset.
Use `mappo_2v2_completion_pretrained.yaml` or
`mappo_2v2_completion_pretrained_shared.yaml` with `--pretrained-actor` to
fine-tune independent actors or a shared actor. The historical PPO baseline
uses `ppo_historical_reproduction.yaml` and stops after 12.288M decisions on
the original 120M-decision learning-rate schedule.

Use `--resume RUN_DIR` to continue from `latest.pt`. For checkpoint evaluation,
pass `--eval --eval-protocol final --checkpoint RUN_DIR` with the training
scenario and its task overrides. The directory selects `best.pt`. The
selection and final protocols use the scenario’s configured maps and seeds.

## Circle PPO pretraining for 20,000 episodes on HPC

`scenarios/ppo_circle_pretrain_20k.yaml` keeps the usual circle-track task,
network, PPO settings, 64 workers × 8 environments, CUDA learning, CPU collection,
and W&B logging. Its budget is 20,000 completed training episodes instead of
120M decisions. The linear learning-rate schedule follows the episode budget.
Parallel collection can finish a few additional episodes in its final batch.

Submit from the repository on the HPC login node:

```bash
sbatch --chdir="$PWD" scripts/hpc/ppo_circle_pretrain_20k.sbatch
```

The job uses the existing reserved partition, QoS, node, and container environment
from the other PPO launchers. Override Slurm resources at submission if required
by the cluster. Checkpoints and logs are saved in
`pretrained_models/circle_ppo_seed42_20k/`. On successful completion, the selected
`best.pt` is atomically copied to `pretrained_models/mlp_base_ppo_seed42_best.pt`.
The run's `final.pt` and resumable `latest.pt` remain in its directory.

To recover the same run after a walltime limit or interruption:

```bash
RESUME=1 sbatch --export=ALL --chdir="$PWD" scripts/hpc/ppo_circle_pretrain_20k.sbatch
```

`REPO_DIR`, `ENV_PATH`, `CRUN`, and `RUN_DIR` may be overridden in the submission
environment. A fresh submission refuses to overwrite an existing run directory.

## Ray Tune sweeps

Run from a clean, reviewed checkout with Ray 2.59 (`ray[tune,default]`) in the
project’s Python environment. Add `--preflight-only` to validate the sweep on
CPU without starting Ray or training. A local cluster starts automatically. Set
`RAY_ADDRESS` to connect to an existing cluster with the same checkout,
environment, and shared output paths on its workers.

```bash
# Local CPU-only smoke sweep
CUDA_VISIBLE_DEVICES='' python scripts/ray_tune_trials.py \
  --sweep-spec scripts/sweeps/architectures_cpu_smoke.yaml

# Local GPU sweep (use 0 for one GPU, or 0,1 for two)
CUDA_VISIBLE_DEVICES=0,1 python scripts/ray_tune_trials.py \
  --sweep-spec scripts/sweeps/ppo_lr_smoke.yaml
```

Submit one Slurm task per node and set the CPUs available to each Ray node
with `--cpus-per-task`. Use a per-node GPU request such as `--gres=gpu:2`, or
omit the GPU request for a CPU-only sweep. These Wahab examples use the same
sweep as the local GPU example. Choose the partition, account and matching QoS
at submission. Set `VENV` for a shared Python environment and `CRUN` when using a
container wrapper. The launcher detects the existing Wahab wrapper when present.

```bash
# One node, two GPUs on the reserved partition
sbatch --partition=reserved-q1wang --account=q1wang --qos=q1wang --time=01:00:00 \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=16 --gres=gpu:2 \
  scripts/hpc/ray_tune.sbatch --sweep-spec scripts/sweeps/ppo_lr_smoke.yaml

# Two nodes, one GPU per node on the public partition
sbatch --partition=gpu --account=q1wang --qos=normal --time=01:00:00 \
  --nodes=2 --ntasks-per-node=1 --cpus-per-task=8 --gres=gpu:1 \
  scripts/hpc/ray_tune.sbatch --sweep-spec scripts/sweeps/ppo_lr_smoke.yaml
```

Tune schedules each trial on one node, up to the sweep’s `max_concurrent_trials`.
For the GPU example, one GPU runs trials sequentially and two GPUs can run two
trials at once when each has six CPUs available. Use `--trial-timeout 0` for
long runs that need to disable the per-attempt timeout. Slurm nodes must have
matching usable CPU and GPU counts. `attempts/<job-id>/allocation.tsv` records the
discovered layout and `gpu_*.csv` captures GPU utilization where available.
The profiling matrix requires its specified layouts.

To recover an interrupted sweep, repeat the command with `--restore`, the same
sweep and experiment ID, the original checkout revision, and the original
absolute output root (`--root` if customized). For Slurm, the `sbatch` resource
options can change as long as one node can fit a trial and any fixed layout is
satisfied. Resume uses the frozen trial configurations, checkpoints, and W&B IDs.

## PPO → MAPPO pipeline

Create the pipeline once from a clean checkout. Creation freezes the PPO and
MAPPO sweeps and verifies the PPO configuration. These commands use the project’s
Python environment from the repository.

```bash
PIPELINE="$PWD/outputs/ppo_mappo_pipeline/matched-run"
python scripts/ppo_mappo_pipeline.py --create --pipeline-id matched-run --root "$PIPELINE"

# Run locally or against an existing Ray cluster selected with RAY_ADDRESS.
python scripts/ppo_mappo_pipeline.py --run --root "$PIPELINE" --stage ppo
python scripts/ppo_mappo_pipeline.py --run --root "$PIPELINE" --stage mappo
```

Omit `--stage` to run all remaining stages locally. PPO needs 20 CPUs and one
GPU per trial. MAPPO needs 10 CPUs and one GPU per trial, and can run up to four
trials concurrently. Four trials on one node need at least 40 usable CPUs and
four GPUs. On one GPU, MAPPO runs its six trials in sequence. Each stage checks
its upstream artifacts before running.

For a Slurm dependency chain, set the resource options independently for each
stage. Use `--dry-run` in place of `--submit` to preview. Add `--stage ppo` or
`--stage mappo` to submit just that stage after its dependencies are verified.
On Wahab, run the same Python command through the project container, for example
`crun.python3 -p "$HOME/envs/marl-race" python3 scripts/ppo_mappo_pipeline.py --dry-run`.

```bash
python scripts/ppo_mappo_pipeline.py --submit --root "$PIPELINE" \
  --sbatch-option ppo=--partition=reserved-q1wang --sbatch-option ppo=--account=q1wang \
  --sbatch-option ppo=--qos=q1wang --sbatch-option ppo=--cpus-per-task=24 \
  --sbatch-option mappo=--partition=gpu --sbatch-option mappo=--account=q1wang \
  --sbatch-option mappo=--qos=normal --sbatch-option mappo=--nodes=2 \
  --sbatch-option mappo=--cpus-per-task=24 --sbatch-option mappo=--gres=gpu:2

# After terminal failure, resume just MAPPO on one GPU.
python scripts/ppo_mappo_pipeline.py --submit --resume "$PIPELINE" --stage mappo \
  --sbatch-option mappo=--partition=gpu --sbatch-option mappo=--account=q1wang \
  --sbatch-option mappo=--qos=normal --sbatch-option mappo=--time=08:00:00

# After training and any previous evaluation jobs have exited, run finals locally.
python scripts/ppo_mappo_pipeline.py --run --stage team --root "$PIPELINE" --eval-device cpu
python scripts/ppo_mappo_pipeline.py --run --stage solo --root "$PIPELINE" --eval-device cpu
```

For local recovery, repeat the original `--run` command with the same root.
For a Slurm chain recovery, omit `--stage` and supply options for the remaining
stages. Final evaluation defaults to a visible CUDA device or CPU and uses the frozen
checkpoints, seeds, and protocols on either device.

The root, checkout revision, scientific settings, sources, trial identities,
and W&B IDs stay fixed. Partition, account, QoS, nodes, available CPUs and GPUs,
walltime, and effective concurrency can change between compatible attempts.
Keep the frozen absolute paths accessible on every host. Historical version 1
manifests require their original revision and fixed Slurm layout.

`summary.json` links per-attempt resource records, including Slurm
`allocation.tsv` and `allocation_timing.tsv`. Cumulative trainer time, Slurm
allocation and batch wall time, and queue delay are reported separately where available.

## Independent single-node GPU experiments

Scenarios own budgets, task choices, and ordered `launch.work_items` (output names
mapped to Hydra override lists). `configs/` owns reusable algorithm, adaptation,
reward, checkpoint, and execution profiles. Slurm alone owns hardware allocation.
Use `scripts/hpc/experiment.sbatch` to start the independent work items
defined by the scenario.

`scenarios/ppo_lora_families.yaml` retains **30,000 episodes** per family,
with no-progress termination disabled. Continual learning defaults
to the pilot profile with independent full-fine-tuning and LoRA conditions;
`PROFILE=full` selects the full scenario budget. Stage transitions, evaluations,
and metrics remain owned by `src/training/continual.py`.

The `single_node_gpu` profile reserves four affinity CPUs per learner and
defaults to four environments per worker. It chooses the largest worker count
that fits CPU affinity and divides the scenario’s PPO batches exactly. Set
`WORKERS`, `ENVS_PER_WORKER`, or `RESERVED_CPUS` to specify the layout.

Examples from the repository on Wahab (submission examples only):

```bash
# Four GPUs, 64 CPUs: straight then s_curve on rank 0; one family on each other rank.
REPO_DIR="$PWD" sbatch --ntasks=4 --cpus-per-task=16 --gpus-per-task=1 \
  scripts/hpc/experiment.sbatch

# Five GPUs, 160 CPUs: one concurrent independent family per task.
REPO_DIR="$PWD" sbatch --ntasks=5 --cpus-per-task=32 --gpus-per-task=1 \
  scripts/hpc/experiment.sbatch

# Two independent continual pilot conditions, each with one GPU and 16 CPUs.
REPO_DIR="$PWD" SCENARIO=scenarios/ppo_continual.yaml PROFILE=pilot sbatch --ntasks=2 --cpus-per-task=16 \
  scripts/hpc/experiment.sbatch

# Resume with fewer GPU tasks; keep enough CPUs/task for each frozen collector.
REPO_DIR="$PWD" RUN_ROOT="$PWD/outputs/ORIGINAL_RUN" RESUME=1 \
  sbatch --ntasks=2 --cpus-per-task=32 scripts/hpc/experiment.sbatch

# Generic launcher, including scratch PPO pretraining without checkpoint export.
REPO_DIR="$PWD" SCENARIO=scenarios/ppo_circle_pretrain_20k.yaml \
  sbatch --ntasks=1 --cpus-per-task=16 scripts/hpc/experiment.sbatch \
  --set execution=single_node_gpu
```

The launcher assigns work items in rank order and freezes every resolved
configuration, source hash, and W&B identity in `launch_manifest.yaml`.
`RESUME=1` reuses the original root and resumes unfinished items from
`latest.pt`, or from the frozen source if an item has no checkpoint.
Ensure the allocation fits each frozen collector. Roots created before the
manifest format need the original scenario and each item’s `launch_config.yaml`.
If the training revision cannot be verified, review compatibility before using
`--allow-code-change`.

Inspect the local assignment with
`python scripts/launch_experiment.py --scenario scenarios/ppo_lora_families.yaml --root /tmp/example --plan`.
`--prepare-only` freezes a real run root without training. Pass repeatable
`--set` flags after the batch script name to override settings on a fresh run.

## Solo PPO adaptation on generated maps

`scenarios/ppo_generated_lora.yaml` trains one shared actor-head adapter against
an immutable PPO base. Hidden linear layers receive low-rank residuals; the
base head and driving encoder stay frozen. The local critic is initialized from
the source and fully trained. Exploration (`log_std`) is trainable by default;
set `adaptation.lora.train_log_std=false` to freeze it. Rank and alpha default to 4.
The existing MAPPO per-agent LoRA preset remains separate.

```bash
# One independent adapter run for each map family, using the same base checkpoint.
for family in straight sweeper hairpin chicane s_curve; do
  python run.py --scenario scenarios/ppo_generated_lora.yaml \
    --checkpoint /absolute/path/to/pretrained/best.pt \
    --set maps=generated_${family} \
    --output-dir outputs/ppo_lora_${family}
done

# Full fine-tuning control across all generated training maps.
python run.py --scenario scenarios/ppo_generated_finetune.yaml \
  --checkpoint /absolute/path/to/pretrained/best.pt \
  --output-dir outputs/ppo_generated_finetune
```

`maps=generated` retains the frozen 40/8/12 train/validation/test split.
Each family preset uses four training maps, one validation map, and one test
map. Training cycles round-robin through training maps; checkpoint selection
uses validation maps. Match the map preset, seeds, budget, and task settings
between adaptation methods. Scenarios default to 4,096,000 decisions, constant
learning rate 0.0001, five target laps, and W&B disabled. These are starting
settings, not tuned budgets.

Saved LoRA checkpoints include the base actor, adapters, critic, and source
identity. Evaluation reconstructs the adapter from checkpoint metadata; the
original source file is not needed to evaluate or resume. Automatic switching
between family adapters and adapter-only exports are not implemented.

```bash
python run.py --scenario scenarios/ppo_generated_lora.yaml \
  --set maps=generated_straight --resume outputs/ppo_lora_straight

python run.py --scenario scenarios/ppo_generated_lora.yaml \
  --set maps=generated_straight --eval --eval-protocol final \
  --checkpoint outputs/ppo_lora_straight \
  --output-dir outputs/ppo_lora_straight_test
```

## Controller rendering and profiling

The MAPPO completion and serial profiling presets use hybrid PP/FTG opponents.
Select Racing MPC with `--set controllers=racing_mpc`. Use the same controller
and overrides across comparison arms.

```bash
python run.py --scenario scenarios/render/hybrid_pp_ftg.yaml
python run.py --scenario scenarios/render/racing_mpc.yaml

PYTHONPATH=src PYGLET_HEADLESS=true python -m application.benchmark_controllers \
  --controllers hybrid_pp_ftg racing_mpc --maps circle_map --seeds 42 \
  --cases solo traffic --max-steps 256 --max-speed 5 --output outputs/controller_throughput.json
```

The controller benchmark reports completion, collisions, and per-controller
and environment timings. For matched comparisons, use the same maps, seeds,
speed limits, and warm-up. The 256-decision example measures throughput.
Earlier quality and timing results are recorded in the Git history of this README.

For a CPU/CUDA throughput comparison, run each device sequentially with detailed
stage and controller timing disabled. These commands use matching settings within
each pair: seed 42, one CPU thread, and 16,384 environment decisions. MAPPO uses
two learners and the shared hybrid opponents. Use fresh output directories for
each repetition.

```bash
for device in cpu cuda; do
  python3 run.py --scenario scenarios/ppo_serial_profile.yaml \
    --set execution.params.device="$device" \
    --set execution.experiment.measure_stages=false \
    --seed 42 --torch-threads 1 --total-steps 16384 \
    --output-dir "outputs/profiling/ppo_${device}_throughput" --quiet
done

for device in cpu cuda; do
  python3 run.py --scenario scenarios/mappo_hybrid_serial_profile.yaml \
    --set execution.params.device="$device" \
    --set execution.experiment.measure_stages=false \
    --set experiment.measure_hybrid=false \
    --seed 42 --torch-threads 1 --total-steps 16384 \
    --output-dir "outputs/profiling/mappo_hybrid_${device}_throughput" --quiet
done

python3 scripts/compare_profile_runs.py \
  --pair PPO outputs/profiling/ppo_cpu_throughput outputs/profiling/ppo_cuda_throughput \
  --pair MAPPO_hybrid outputs/profiling/mappo_hybrid_cpu_throughput outputs/profiling/mappo_hybrid_cuda_throughput \
  --output outputs/profiling/throughput_comparison
```

`compare_profile_runs.py` checks matching workloads and actual devices. It
reads `run_summary.json`, `update_metrics.csv`, and `config_snapshot.json`
and writes paired throughput and workload comparisons as CSV and JSON.

For detailed hybrid timing, run `scenarios/mappo_hybrid_serial_profile.yaml`.
For detailed MPC timing, use the serial MAPPO preset with the MPC controller.

```bash
python run.py --scenario scenarios/mappo_serial_profile.yaml \
  --set controllers=racing_mpc --set experiment.measure_mpc=true \
  --set execution.experiment.measure_stages=true \
  --output-dir outputs/mappo_mpc_profile
```

`run_summary.json` and `update_metrics.csv` record stage and controller timing.
Keep workloads, devices, and warm-up settings fixed when measuring profiler
overhead. See the [P1](reports/physics_p1.md) and [P2](reports/physics_p2.md)
reports for physics and collection benchmarks and reproduction commands.

## Slurm scaling matrix

```bash
bash scripts/hpc/submit_ppo_matrix.sh --dry-run
# On Slurm, use --submit to enqueue the validated profiles in order.
python scripts/hpc/summarize_ppo_matrix.py /path/to/matrix/jobs.tsv
```

The matrix preflights each profile against its fixed node layout and records
the profiles and job IDs in `jobs.tsv`. Use `--suite nodes` or
`--suite workers` to select a subset.
