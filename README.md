# MARL racing

Reinforcement learning experiments for racing.

Physics baseline for the F0–F2 geometry correction: `origin/multiNode` at
`7d65c76fadff9081ce59c4a42c7400b892adc92e`. Default dynamics remain
`combined_slip_st` v2 / MF6.1, RK4 with 0.001 s maximum substeps and a 0.05 s
simulation step (standalone simulator default: 0.01 s). Its chassis is 0.58 ×
0.31 m, with axles at +0.15875 / −0.17145 m; the legacy preset keeps its
0.32 × 0.225 m footprint and v1 dynamics. Run provenance records the independent
`vehicle_geometry_collision` behavior version (2); dynamics/configs are unchanged.

# 1. Solo PPO pretraining
python run.py --scenario scenarios/ppo_lap_completion_pretrain.yaml \
  --output-dir outputs/pretrain

# 2. PPO track transfer: circle
python run.py --scenario scenarios/ppo_lap_completion_transfer.yaml \
  --checkpoint outputs/pretrain/best.pt \
  --output-dir outputs/transfer_circle

# 3. PPO track transfer: Budapest, three laps
python run.py --scenario scenarios/ppo_lap_completion_transfer_3lap.yaml \
  --checkpoint outputs/pretrain/best.pt \
  --output-dir outputs/transfer_budapest

# 4. MAPPO from scratch
python run.py --scenario scenarios/mappo_2v2_completion_scratch.yaml \
  --output-dir outputs/mappo_scratch

# 5. MAPPO: independent pretrained actors
python run.py --scenario scenarios/mappo_2v2_completion_pretrained.yaml \
  --pretrained-actor outputs/pretrain/best.pt \
  --output-dir outputs/mappo_pretrained

# 6. MAPPO: shared pretrained actor
python run.py --scenario scenarios/mappo_2v2_completion_pretrained_shared.yaml \
  --pretrained-actor outputs/pretrain/best.pt \
  --output-dir outputs/mappo_shared

# 7. MAPPO: per-agent LoRA
python run.py --scenario scenarios/mappo_2v2_completion_lora.yaml \
  --pretrained-actor outputs/pretrain/best.pt \
  --output-dir outputs/mappo_lora

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

## Concurrent HPC LoRA family training: 20,000 episodes each

`scenarios/ppo_generated_lora_20k.yaml` uses one independent adapter for each of
straight, sweeper, hairpin, chicane, and S-curve tracks. Every family trains for
20,000 completed episodes from the same frozen circle PPO checkpoint. Each run
keeps 64 CPU collector workers × 8 environments, CUDA learning, CPU collection,
the historical PPO rollout/update settings, rank/alpha 4, and the adaptation
learning rate of 0.0001. W&B logging is enabled. Training-time evaluation is
disabled; no preflight or pilot runs are submitted. Normal compatibility checks
still run when the trainer loads its checkpoint.

From the HPC repository, submit:

```bash
sbatch --chdir="$PWD" scripts/hpc/ppo_lora_families_20k.sbatch
```

Default resources are two nodes and five GPU tasks, with at most four tasks per
node. Each node has 32 CPUs and four V100 GPUs; each task receives one GPU and
eight CPUs. All five map-family adapters run concurrently, using five GPUs total
across the two nodes. The remaining three GPUs are not requested. These are
separate policies and optimizers, without cross-GPU gradient sharing.

`--ntasks-per-node=4` is the per-node maximum with the explicit five-task budget.
The launcher uses `--mem=0` to request each allocated node's configured memory
rather than assume a 256 GB capacity. The historical collector layout remains
64 workers × 8 environments per adapter; these workers share the adapter's eight
allocated CPUs, so CPU collection is oversubscribed. The collector layout has not
been retuned. PPO settings and episode budgets are unchanged.

The default walltime is four hours on the existing public GPU
partition/account/QoS. Change `--time` only to a limit supported by the selected
partition/QoS. The standard task settings are five target laps, up to 80,000 steps
per episode, with track-limit termination. Parallel collection may overshoot the
episode target in its final batch.

The source defaults to `pretrained_models/mlp_base_ppo_seed42_best.pt` and is
copied once to `outputs/ppo_lora_20k_JOBID/source.pt`. Override
`SOURCE_CHECKPOINT` and/or `RUN_ROOT` in the submission environment if needed.
Each family directory contains `train.log`, native CSV/physics logs, W&B
provenance, `latest.pt`, and `final.pt` (base + adapter + critic). Evaluation is
disabled, so there is no selected `best.pt`. Checkpoints stay separate from the
original pretrained model.

Recover unfinished families after interruption or walltime expiry:

```bash
RUN_ROOT="$PWD/outputs/ppo_lora_20k_JOBID" RESUME=1 \
  sbatch --export=ALL --chdir="$PWD" scripts/hpc/ppo_lora_families_20k.sbatch
```

Resume retains the frozen source snapshot and skips completed families. Families
with `latest.pt` recover from that checkpoint; any family with no recovery
checkpoint restarts from the base. Keep code, training settings, seeds, dataset,
and output location unchanged when resuming. Concurrency can be changed because
the family jobs are independent. `REPO_DIR`, `ENV_PATH`, and `CRUN` use the same
override conventions as the other HPC launchers.

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

Before a research run, validate the actual pretrained checkpoint against the
current network, physics, action, and observation contracts, and assess generated
track drivability. Geometric validation alone does not establish drivability.
A short run with a synthetic source checks integration, not learned performance.

PPO pretraining and all four MAPPO completion scenarios share the historical PPO
training setup in `configs/training/historical_ppo.yaml`: 64 workers with 8
environments each, 4,096 collector frames per batch, 409,600 learner samples per
update, 10 epochs, minibatches of 1,024, and a linear learning-rate schedule from
0.001 to 0.0001 over 120M environment decisions. Episodes allow 80,000 steps;
checkpoints are saved every 1,024,000 decisions. Learning runs on CUDA and
collection runs on CPU. These scenarios keep their existing rewards and run the
full budget; the historical reproduction alone keeps its 12.288M diagnostic stop.
Serial profiling scenarios use their separate, smaller workloads.

The MAPPO scratch, pretrained, shared pretrained, and LoRA scenarios use identical
`hybrid_pp_ftg` opponents. They run Pure Pursuit by default, with speed-dependent
lookahead, steering-lag compensation, curvature/grip speed limits, and LiDAR gap
avoidance. Gap checks include vehicle width, nearby traffic, and stationary cars.
Five clear decisions are required before returning to Pure Pursuit. Steering is
filtered and rate limited; acceleration is smoothed, with immediate safety braking.
Commands are physical `[steering radians, target m/s]` before the existing
`WheelReferenceAdapter` converts speed to wheel radians/second.

The shared settings are in `configs/controllers/hybrid_pp_ftg.yaml`. The default
opponent cap is **8 m/s**, with a **3 m/s cap during avoidance**. Curvature, grip,
and stability checks reduce the target speed when needed. The environment's
8 m/s limit is a ceiling; lower opponent settings are preserved. To change speed:

```bash
python3 run.py --scenario scenarios/mappo_2v2_completion_scratch.yaml \
  --set controllers.hybrid_pp_ftg.max_speed=4.0
```

Use the same controller overrides for every comparison arm. Lower the cap when
evaluating more conservative opponent settings. To select
MPC instead, add `--set controllers=racing_mpc`. The existing
`scenarios/mappo_serial_profile.yaml` explicitly keeps the 8 m/s MPC baseline.
To view the hybrid controller alone:

```bash
python3 run.py --scenario scenarios/render/hybrid_pp_ftg.yaml
```

Evaluate without training, policy inference, rendering, or online logging:

```bash
PYTHONPATH=src PYGLET_HEADLESS=true python3 -m application.benchmark_controllers \
  --maps circle_map Budapest_map --seeds 42 43 44 \
  --cases solo traffic stationary --max-speed 8 --output outputs/controller_quality_speed8.json

PYTHONPATH=src PYGLET_HEADLESS=true python3 -m application.benchmark_controllers \
  --controllers hybrid_pp_ftg racing_mpc --maps circle_map --seeds 42 \
  --cases solo traffic --max-steps 256 --max-speed 5 --output outputs/controller_throughput.json
```

The quality report records per-car lap completion, collision termination, speed,
lateral motion, yaw rate, cross-track error, steering changes, FTG use, mode
switches, and controller exceptions. `stationary` places a parked car 8 m ahead;
`traffic` uses four controllers with the same seeded spawn procedure. The runner
uses `RaceTask` with every car assigned a fixed controller and no policy owners.
Controller timing includes the wheel adapter. Environment throughput includes
`RaceTask.step` and snapshots; report aggregation is excluded. Physics/LiDAR JIT
is primed before a seeded reset; the first 20 decisions are excluded from steady
throughput and controller mean/p95. Use equal maps, seeds, speeds, step budgets,
and warm-up counts, and run comparisons without competing workloads. A 256-step
throughput run is too short to assess lap completion.

Earlier validation on 2026-10-11 at a 5 m/s cap completed clean solo laps on
all nine racing tracks. Circle/Budapest tests with seeds 42–44 completed all
36 car runs across solo, four-car traffic, and parked-car cases without collisions
or controller exceptions. Three-lap solo and parked-car checks passed on both
maps. MPC also completed the matched 5 m/s solo reference laps without collisions:
mean speeds were 4.69/4.72 m/s (circle/Budapest), versus 4.92/4.45 m/s for the hybrid.
The initial 8 m/s solo setting also completed both maps.
A 32-decision seeded `RaceTask` replay matched state arrays exactly.

After raising the shared cap to 8 m/s, solo laps passed on all nine tracks at
seed 42. Solo, four-car traffic, and parked-car cases across those tracks, plus
Circle/Budapest seeds 43–44, completed **77 of 78 car runs**, with **zero collisions
or controller exceptions**. The Spielberg parked-car case stalled until the
6,000-decision limit; the same case also stalled at 5 m/s. This remains an obstacle
avoidance limitation. Circle/Budapest passed all 36 car runs across three seeds.
Mean completed-run speeds ranged from 4.62 to 7.72 m/s; an 8 m/s cap still permits
slower cornering and avoidance. The regression suite passed 263 tests and skipped
two GPU tests; the known PPO startup test still fails when run separately.
A 256-decision MAPPO smoke run recorded 512 profiled hybrid calls, including PP,
FTG, warm-up, and steady phases. Reports are in
`outputs/controller_speed8_{all_tracks,extra_seeds,spielberg_comparison}.json`,
with `outputs/controller_speed8_evaluation_summary.json`. These quality checks
ran alongside other checks; their timing samples are not an isolated throughput
comparison.

The isolated 256-decision circle benchmark at a 5 m/s cap measured:

| Case | Hybrid steps/s | MPC steps/s | Hybrid controller ms | MPC controller ms |
| --- | ---: | ---: | ---: | ---: |
| Solo | 556.3 | 48.7 | 0.64 | 19.12 |
| Four-car traffic | 113.0 | 10.1 | 1.18 | 23.61 |

These are steady simulation timings, with means per controller call and no
learner workload. They show about 11× higher environment throughput and 20–30×
lower controller cost on this machine. Timing results will vary by host. Reports
are saved under `outputs/controller_{tracks,quality,endurance,speed8,throughput}.json`,
with a compact `outputs/controller_evaluation_summary.json`. The test run passed
247 tests and skipped two GPU tests. The existing fresh-process PPO startup test
fails because it checks `trainer.env` before `train()` initializes it.

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

The collector verifies matching effective configurations, completed decision
budgets, disabled detailed/controller profiling, and actual policy/update devices
(so a CUDA fallback is rejected). It writes CSV and JSON reports with decisions/s,
active learner samples/s, optimizer steps/s, elapsed time, coarse collection/update
time, and workload counters. Source data remain in `run_summary.json`,
`update_metrics.csv`, and `config_snapshot.json` in each run directory. Coarse
update timing still synchronizes CUDA at update boundaries; detailed per-stage
synchronization is disabled. Elapsed time includes trainer initialization and
in-run hooks, but excludes application setup and final artifact saving. Identical
seeds do not guarantee identical CPU/CUDA trajectories. Compare devices within
each pair; PPO and MAPPO have different workloads.

Hybrid profiling is optional (`experiment.measure_hybrid=false` by default).
The dedicated serial preset enables internal hybrid and environment stage timing:

```bash
python3 run.py --scenario scenarios/mappo_hybrid_serial_profile.yaml \
  --output-dir outputs/mappo_hybrid_profile
```

It keeps the shared 8 m/s hybrid settings, fixed grip, seed 42, CPU execution,
and the same 16,384-decision MAPPO budget/update settings as the MPC serial preset.
Both serial presets use an 8 m/s opponent cap.
For a baseline measuring profiling overhead, use the identical command with
`--set experiment.measure_hybrid=false` and a separate output directory.
On other scenarios, enable internal timing with `--set experiment.measure_hybrid=true`;
environment stage timing is independent.

`run_summary.json` includes `hybrid_profile`. Its exclusive stage breakdown covers
map preparation, Pure Pursuit, curvature/stability speed planning, LiDAR
preprocessing, traffic prediction, path checks/hysteresis, gap selection, smoothing,
and other bookkeeping. Each stage reports total seconds, invocation count, mean
milliseconds per controller call and per stage invocation, and percentage of
hybrid time. Gap selection runs only when needed; Pure Pursuit is computed even
in FTG mode. Total call timings include mean, maximum, and standard deviation.

The report partitions calls by opponent, initialization/warm-up/steady phase,
PP/FTG/invalid mode, active vehicles/learners/hybrid opponents, nearby traffic,
blocked path, emergency braking, and episode/map startup. Counters include mode
switches, braking/stopped commands, missing scans, invalid state, and failed gap
searches. Mean input/target speed, absolute steering, LiDAR hits, preview samples,
and tested gap candidates help explain workload and behavior. Mode fractions
and rates use controller calls, including hysteresis decisions. Active counts use
the physical lifecycle; traffic can include retired stationary wrecks.

`update_metrics.csv` exports cumulative `perf/hybrid/...` scalars and stage metrics
for overall, per-agent, phase, and mode partitions. No per-decision files or timing
samples are retained. Totals reset per `train()` call and survive episode resets;
callbacks are restored when training ends. Disabled hybrid profiling reads no
controller clock. Internal seconds exclude the wheel adapter and aggregation;
the outer `fixed_controller` stage includes both. Context partitions overlap and
must not be added together. The first completed update is warm-up by default;
use the full preset for representative steady-state measurements. A short smoke
run can add `--total-steps 128 --set algorithm.params.n_steps=32
--set algorithm.params.frames_per_batch=16` to exercise both phases.

MPC profiling is optional (`experiment.measure_mpc=false` by default). Run the
existing serial MAPPO preset with internal MPC and environment stage timing:

```bash
python3 run.py --scenario scenarios/mappo_serial_profile.yaml \
  --set experiment.measure_mpc=true \
  --set execution.experiment.measure_stages=true \
  --output-dir outputs/mappo_mpc_profile
```

For a matched throughput baseline, use the same command with
`experiment.measure_mpc=false` and a different output directory. Keep the seed,
step budget, device, stage timing, and warm-up count identical. The preset's
16,384 decisions are intended for a later profiling experiment. A minimal smoke
run can add `--total-steps 9`; it will generally have no steady-state samples
with the preset's rollout length. For PPO use `scenarios/ppo_serial_profile.yaml`;
that solo scenario has no MPC opponents.

`run_summary.json` contains `mpc_profile`, with overall totals and independent
partitions by agent (including `car_2` and `car_3`), phase, active physical vehicles
(1–4), active learners, active MPC opponents, nearby traffic, optimization starts,
warm-plan use, first episode decision, and new Numba shooting signatures.
`update_metrics.csv` also exports cumulative `perf/mpc/...` scalar metrics.
No per-decision or per-optimization files are written.

Stage seconds are exclusive: `scipy_optimization` subtracts timed shooting
objectives from total solver wall time. `solver_seconds` remains an inclusive
diagnostic and must not be added to the stage breakdown. `other` accounts for
bookkeeping. Solver evaluations and iterations come from SciPy's `nfev` and
`nit`; failures include budget exhaustion. Braking selection and infeasible-plan
fallback have separate rates. Mean stage times divide by MPC calls; solver means
divide by optimization starts; objective means divide by `nfev`. MPC milliseconds
per environment decision include decisions without active MPC opponents. MPC's
end-to-end percentage uses trainer elapsed time, including updates and hooks,
excluding application setup. The existing `fixed_controller` timing includes
adapter conversion and profiler aggregation, while internal MPC seconds exclude
them. Context partitions overlap each other and must not be added together.

Totals reset at each `train()` invocation and survive episode resets. Phase
boundaries follow `profiling_warmup_updates` (default 1, included in the budget),
even when stage timing is disabled. First decisions and calls loading new Numba
signatures help identify map/JIT startup costs; compilation/cache-load time stays
inside the calling stage and is not measured separately. Active counts use the
physical lifecycle; nearby traffic can include retired stationary wrecks.

Disabled profiling installs no MPC callback and leaves the original objective
untimed. Enabled profiling adds two clock reads and one scalar accumulation per
objective evaluation, then aggregates after the solver. Compare matched runs to
measure overhead; short timings are noisy. The Numba model, caching, solver,
limits, controller actions, and learning configurations are unchanged.

Environment `physics`, `lidar`, `observation_reward`, and `fixed_controller`
stages separate simulation from controller costs. Hybrid runs use the same
`fixed_controller` stage timing and opt-in `hybrid_profile`; `mpc_profile` only
measures Racing MPC instances.
Prediction, footprint, and traffic collision costs within MPC's compiled `_shoot()`
kernel remain combined; timing inside that kernel would require further instrumentation.
