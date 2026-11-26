# Hierarchical racing implementation and HPC handoff

Branch: `hierarchical-learning`, based on local `main` at `c3446b8`.
No remote/HPC connection, SLURM submission, Ray cluster, full training experiment,
or specialization matrix was started. Existing pretrained checkpoints were not changed.

## Implementation status

| Phase | Result | Commit |
|---|---|---|
| 1 | Hash-pinned, provenance-validated six-actor frozen bank, actor-only reconstruction and independent batched routing | `b9856a8` |
| 2 | Fixed/random/geometry selectors and per-environment selection/switch state | `7d726dd` |
| 3 | Solo macro-task, separate integer categorical action specification, unchanged LL observations and physics | `1c01794` |
| 4 | Categorical HL PPO, independent critic, duration-aware GAE and checkpoint references | `835e9a6` |
| 5 | Existing serial/process collectors, asynchronous checkpoint selection, configuration and launcher/Tune planning | `61a4834` |
| 6 | Composite baselines, research inventory, offline CSV/SVG reporting and switch diagnostics | `59664dc` |

The following handoff commit contains this report and final review corrections.
Use `git log -7 --oneline` to identify it. **No tests, training, configuration
execution or sweeps were run after the user requested that remaining validation
be deferred to the HPC.** Final phase-six/reporting/configuration corrections
therefore await HPC validation.

### Implemented and locally exercised before that request

Actual checkpoint bank loading; deterministic parity against standalone policies;
independent routing and repeated switching; fixed, random and geometry selectors;
macro rewards and actual durations; K=1/5/10/20 GAE; terminal/truncation masks;
LL freezing; short CPU PPO updates; serial and two-worker collection;
checkpoint save/load/resume; standalone learned-checkpoint evaluation;
minimal asynchronous validation and best-checkpoint selection; launcher planning;
four-trial **non-executing** Ray configuration preflight.

The native checkpoint schema and existing PPO algorithm name are retained.
HRL is an isolated PPO mode, selected by `hierarchical.enabled`, with explicit
categorical distribution/action contracts. The bank uses six preloaded residual
actors, not merged weights or optimized shared-base routing. A weak process-local
cache shares actors across environments; selection state stays with each task.
Only HL actor/critic parameters enter the optimizer or weight synchronization.

Rewards are `sum(gamma_ll**i * reward_i)`, bootstrap discounts are `gamma_ll**k`,
and GAE traces are `(gamma_ll * lambda_ll)**k`. Done stops GAE recurrence;
termination additionally removes final-state bootstrap. Collector boundaries
bootstrap without inventing episode endings. LL decisions remain the primary
budget; `n_steps` and rollout/minibatch sizes count HL samples. Batched collection
can overshoot the LL budget; compare recorded actual decisions and overshoot.

### Newly modified and awaiting HPC validation

The v2 HL observation replaces raw dwell counts with `tanh(elapsed_seconds / scale)`
(default scale 1 s), measured from actual task durations. It optionally appends
signed curvature then width samples at an independently configured spacing using
cached LL preview geometry. The LL observation is unchanged. V1 HL checkpoints
are incompatible even when dimensions agree. Raw counts and durations remain
available for diagnostics. Preview defaults off; enabled scales default to the
existing physical 1/m and m normalizers of 1.0.

PPO still receives internally discounted macro rewards and duration-aware GAE.
Historical `episode/reward`, `training_return`, evaluation `mean_episode_reward`,
reward components and agent reward fields remain sums of those macro rewards for
hierarchical runs. Use `hierarchical/return_ll_undiscounted` (or evaluation
`mean_return_ll_undiscounted`) for conventional PPO/LoRA comparisons. Also record
`return_macro_discounted_sum` and `return_ll_globally_discounted`; discounting is
in LL-decision units. Existing `distance_m` diagnostics are legacy aliases for
positive centerline progress, not physical distance. New metrics explicitly
report net progress, accumulated forward progress and sampled-physics polyline
distance in metres. Physical distance is a numerical path-length approximation.

Selector streams are independent of map/spawn streams and derive from each
environment seed. Geometry fallback reasons, actual switch windows, elapsed
simulation time, inference timings and bank tensor storage are reported.
Update CSV rows contain actual LL decisions per PPO update and a JSON array of
actual LL decisions for each collection batch leading to that update. Batch
rounding, short terminal macros and the final budget can change these values.
These corrections and focused tests have not been runtime validated.

### Implemented but not experimentally validated

Final configuration/provenance corrections, conventional composite baseline
presets, sequence-selector switch studies, extended offline reports, and the
final parameter/accounting diagnostics have not been rerun locally. Full racing,
specialization, switching, held-out generalization, profiling, GPU/DDP and actual
Ray execution remain deferred. Their results are not established by smoke tests.

### Not implemented or limited

- **Shared multi-node learner/collection is not supported by the existing backend.**
  `training/distributed.py` explicitly supports one node and at most two local
  CUDA ranks. Independent Ray trials can span nodes; that is a different workload.
  There is no valid shared multi-node HRL command to provide without extending
  that backend. This limitation is not presented as a completed HPC check.
- Optional learned-policy categorical probabilities are not recorded; adding them
  without duplicated inference requires evaluator policy-output interface work.
- No exact simulator-state restoration. Switch-pair studies use identical seeded
  starts/spawn schedules. The sequence selector runs an approach and then holds
  its final skill; follow-up measurements are limited to recorded LL decisions.
- Segment timing has LL-decision resolution. First/last visits may be partial;
  reports flag these. Selection distance to a matching skill's next entry is a
  periodic diagnostic, not a label-matching objective or optimal oracle.
- Trajectories record actions, physical command changes, pose, speed, lateral
  error, switches and geometry. HL probability trajectories are not recorded.
- Existing learner/collector timing and worker-memory reporting are reused;
  comprehensive per-skill profiling and performance conclusions await HPC work.
- Comparative learning curves use existing `update_metrics.csv` and
  `evaluation_history.jsonl`; no new plotting framework is introduced.
- Full upstream training cost cannot be recovered from selected `best.pt` files
  alone. Combine original base/family run summaries with incremental HL cost;
  do not count only HL updates or equate best-checkpoint progress with full run cost.

## Checkpoints and dataset

`configs/hierarchical/bank.yaml` is the authoritative ordered, SHA-256-pinned
manifest. Paths are relative to the repository, independent of the HPC source
paths recorded in checkpoint metadata.

| ID | Skill | Required checkpoint |
|---|---|---|
| 0 | base | `pretrained_models/circle_ppo_seed42_20k/best.pt` |
| 1 | straight | `pretrained_models/straight_best.pt` |
| 2 | sweeper | `pretrained_models/sweep_best.pt` |
| 3 | hairpin | `pretrained_models/hairpin_best.pt` |
| 4 | chicane | `pretrained_models/chicane_best.pt` |
| 5 | s_curve | `pretrained_models/s_best.pt` |

All five source identities reference base hash
`8665b07cd09fd5d306fbf0a74d18067a49aafd6c09e5b48a17e68c995a0118cc`.
`mlp_base_ppo_seed42_best.pt` is byte-identical to this base. Metadata confirms
four intended specialist training maps plus a separate specialist validation map
for each adapter, rank=4, alpha=4, shared LoRA actor-head adaptation. Frozen base
weights, source hashes, network, observation/action/physics contracts and LoRA
layer layouts are validated. No missing checkpoint was found.

The inspected Phase5 V2 manifest has, per specialist family, 4 train / 1 validation /
1 test maps, plus 20 / 3 / 7 composite maps. `configs/maps/composite.yaml` selects
only the composite splits, with round-robin episode scheduling. Geometry labels
are available only to the privileged baseline and optional diagnostic records.
They and map identity do not enter learned observations.

Bank accounting observed locally: 660,592 frozen parameters, zero trainable,
2,643,904 bytes of parameter/buffer tensors. HL actor and critic at [128,128]
have 38,790 and 38,145 parameters, respectively (76,935 trainable total).
Tensor storage excludes Python, checkpoint deserialization, simulator and worker RSS.

## Earlier local evidence

CPU-only, one thread per process, rendering disabled, W&B disabled:

- Focused HRL tests and PPO/MAPPO/LoRA/native-env/checkpoint/launcher regressions:
  **228 passed, 9 skipped, 4 warnings in 154.67 s**. The final review edits are newer
  than this result. Skips are not claimed as passed GPU/distributed validation.
- Resumed serial/async run: 21 actual LL decisions, 6 HL samples, 3 optimizer
  updates, 3 completed short episodes; three validation records preserved
  snapshot identity and best selection. Final resumed segment: 0.9925 s recorded
  runtime. Earlier initial-process setup is not included in that segment timing.
- Two-worker final smoke: 28 LL decisions, 8 HL samples, 2 optimizer updates,
  4 short episodes, 11.2291 s recorded runtime. Requested budget was 20; collector
  batching explains the eight-decision overshoot.
- Learned standalone evaluation completed a deliberately seven-step episode.
  Such truncated episodes establish execution only, not racing performance.
- Launcher plan resolved two workers and one environment per worker.
- Ray `--preflight-only` resolved four interval trials without starting Ray.

These are functional observations under laptop resource contention, **not HPC
benchmarks**. A broader regression invocation included execution-oriented mocked
Tune tests; it was interrupted and excluded from the final regression gate.
No real Ray cluster or sweep was started. After memory pressure, remaining
validation workers were terminated; no workloads should be running from this task.

Earlier regression command (recorded, not rerun for this handoff):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  tests/test_hierarchical.py tests/test_torchrl_ppo.py tests/test_ppo_lora.py \
  tests/test_torchrl_mappo.py tests/test_native_torchrl_env.py \
  tests/test_checkpoints.py tests/test_experiment_launcher.py -q --disable-warnings
```

## Commands for HPC execution later

Run from this branch's repository root with the existing HPC environment activated.
Use a fresh output directory for each new run. Commands below are **prepared,
not executed**. CPU/GPU resources must come from an appropriate allocation.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 PYGLET_HEADLESS=true WANDB_MODE=disabled
SCENARIO=scenarios/ppo_hierarchical.yaml
BASE=pretrained_models/circle_ppo_seed42_20k/best.pt
COMMON=(--scenario "$SCENARIO" --no-wandb --no-render --torch-threads 1)
SMOKE=(--total-steps 20 --max-steps 7
  --set algorithm.params.n_steps=4 --set algorithm.params.batch_size=4
  --set algorithm.params.n_epochs=1 --set algorithm.params.frames_per_batch=4)
```

Bank/parity gate and configuration resolution:

```bash
python -m pytest tests/test_hierarchical.py -k 'real_bank or standalone_checkpoint or changed_base' -q
python run.py "${COMMON[@]}" --resolve-config
python run.py --scenario scenarios/ppo_composite_scratch.yaml --resolve-config
python run.py --scenario scenarios/ppo_composite_lora.yaml --resolve-config
```

Fixed, random, geometry and learned checkpoint evaluation:

```bash
python run.py "${COMMON[@]}" --eval --checkpoint "$BASE" --eval-protocol selection \
  --set hierarchical.selector.type=fixed --set hierarchical.selector.skill=1 \
  --set hierarchical.record_trajectory=true --output-dir outputs/hrl/fixed1
python run.py "${COMMON[@]}" --eval --checkpoint "$BASE" --eval-protocol selection \
  --set hierarchical.selector.type=random --output-dir outputs/hrl/random
python run.py "${COMMON[@]}" --eval --checkpoint "$BASE" --eval-protocol selection \
  --set hierarchical.selector.type=geometry --set hierarchical.selector.anticipation=1.0 \
  --output-dir outputs/hrl/geometry
python run.py "${COMMON[@]}" --eval --checkpoint outputs/hrl/train/best.pt \
  --eval-protocol selection --set hierarchical.record_trajectory=true \
  --output-dir outputs/hrl/learned-validation
```

Small serial, multiprocessing, asynchronous and resume checks on the HPC:

```bash
python run.py "${COMMON[@]}" "${SMOKE[@]}" --set evaluation.enabled=false \
  --output-dir outputs/hrl/smoke-serial
python run.py "${COMMON[@]}" "${SMOKE[@]}" --set experiment.parallel_envs=2 \
  --set evaluation.enabled=false --output-dir outputs/hrl/smoke-process
python run.py "${COMMON[@]}" "${SMOKE[@]}" --set experiment.async_evaluation=true \
  --set evaluation.episodes=1 --set evaluation.every_steps=5 \
  --set +experiment.stop_after_steps=7 --output-dir outputs/hrl/smoke-resume
python run.py "${COMMON[@]}" "${SMOKE[@]}" --set experiment.async_evaluation=true \
  --set evaluation.episodes=1 --set evaluation.every_steps=5 \
  --resume outputs/hrl/smoke-resume/latest.pt --output-dir outputs/hrl/smoke-resume
```

Launcher planning without submission, then full composite training when authorized:

```bash
python scripts/launch_experiment.py --scenario "$SCENARIO" \
  --root outputs/hrl/planned --workers 2 --envs-per-worker 1 --plan
python scripts/launch_experiment.py --scenario "$SCENARIO" \
  --root outputs/hrl/train-launch --workers 4 --envs-per-worker 1
# The launcher's default work item is train: checkpoint is train-launch/train/best.pt.
# Direct equivalent, used by the evaluation example above:
python run.py "${COMMON[@]}" --total-steps 4096000 \
  --set experiment.parallel_envs=4 --output-dir outputs/hrl/train
```

B9 and B10 reuse conventional PPO/LoRA with the same composite splits and reward:

```bash
python run.py --scenario scenarios/ppo_composite_scratch.yaml --no-render --no-wandb \
  --total-steps 4096000 --output-dir outputs/hrl/B9
python run.py --scenario scenarios/ppo_composite_lora.yaml --no-render --no-wandb \
  --total-steps 4096000 --output-dir outputs/hrl/B10
```

Specialization matrix (five specialist validation families plus three composite
validation maps); held-out tests are separate and explicitly requested:

```bash
for skill in 0 1 2 3 4 5; do
  python run.py "${COMMON[@]}" --eval --checkpoint "$BASE" --eval-protocol selection \
    --set maps=generated --set evaluation.episodes=8 \
    --set hierarchical.selector.type=fixed --set hierarchical.selector.skill="$skill" \
    --set hierarchical.record_trajectory=true --output-dir "outputs/hrl/matrix/$skill"
done
# Only after model selection is complete: all twelve held-out generated maps.
# Repeat for each frozen skill if explicitly running the final specialization matrix.
python run.py "${COMMON[@]}" --eval --checkpoint "$BASE" --eval-protocol final \
  --set maps=generated --set evaluation.final_test.episodes=12 \
  --set hierarchical.selector.type=fixed --set hierarchical.selector.skill=0 \
  --output-dir outputs/hrl/matrix-test/base
```

B6 best-fixed selection uses **composite validation only**, never the test matrix:

```bash
for skill in 0 1 2 3 4 5; do
  python run.py "${COMMON[@]}" --eval --checkpoint "$BASE" --eval-protocol selection \
    --set hierarchical.selector.type=fixed --set hierarchical.selector.skill="$skill" \
    --output-dir "outputs/hrl/fixed-validation/$skill"
done
PYTHONPATH=src python -m hierarchical.reporting --best-fixed \
  outputs/hrl/fixed-validation/*/evaluation_report.json --output outputs/hrl/B6-selection
# Read B6-selection/best_fixed.json and use its integer skill for subsequent evaluation.
```

Switch compatibility: identical seeds/spawns, source for 20 LL decisions, then
target retained for the remainder. No exact shared-state claim is made.

```bash
for source in 0 1 2 3 4 5; do
  for target in 0 1 2 3 4 5; do
    [ "$source" = "$target" ] && continue
    python run.py "${COMMON[@]}" --eval --checkpoint "$BASE" --eval-protocol selection \
      --set hierarchical.interval=20 --set hierarchical.selector.type=sequence \
      --set "hierarchical.selector.sequence=[$source,$target]" \
      --set hierarchical.record_trajectory=true \
      --output-dir "outputs/hrl/switches/${source}_${target}"
  done
done
PYTHONPATH=src python -m hierarchical.reporting \
  outputs/hrl/switches/*/evaluation_report.json --output outputs/hrl/switch-report
PYTHONPATH=src python -m hierarchical.reporting \
  outputs/hrl/matrix/*/evaluation_report.json --output outputs/hrl/matrix-report
```

H1 held-out composite evaluation, reserved until final evaluation:

```bash
python run.py "${COMMON[@]}" --eval --checkpoint outputs/hrl/train/best.pt \
  --eval-protocol final --output-dir outputs/hrl/held-out
# External evaluation on original maps, excluding the seen circle pretraining map.
python run.py "${COMMON[@]}" --eval --checkpoint outputs/hrl/train/best.pt \
  --eval-protocol final --set maps=solo_transfer \
  --set 'maps.final_test=[L_map,Budapest_map,Hockenheim_map,Melbourne_map,Montreal_map,Shanghai_map,Silverstone_map,Spa_map,Spielberg_map]' \
  --set evaluation.final_test.episodes=9 --output-dir outputs/hrl/external
# Do not use these external results for checkpoint selection or tuning.
```

Single-node profiling/scaling: run each command in an HPC allocation, preserving
LL budgets. These are requested deferred checks, not laptop benchmarks.

```bash
for workers in 1 2 4 8 16 32; do
  # Choose one candidate after inspecting the allocation; do not run a full matrix.
  envs=1; threads=1
  python run.py --scenario "$SCENARIO" --no-wandb --no-render --total-steps 65536 \
    --set experiment.parallel_envs="$workers" --set experiment.envs_per_worker="$envs" --torch-threads "$threads" --set experiment.measure_stages=true \
    --set experiment.measure_services=true --set evaluation.enabled=false \
    --set algorithm.params.n_steps=1024 --set algorithm.params.batch_size=128 \
    --set algorithm.params.frames_per_batch=256 \
    --output-dir "outputs/hrl/scaling-cpu/$workers"
done
# Repeat an identical configuration with a GPU learner; frozen LL inference stays CPU.
python run.py "${COMMON[@]}" --total-steps 65536 --set execution.params.device=cuda \
  --set execution.params.collection_device=cpu --set experiment.parallel_envs=4 \
  --set experiment.measure_stages=true --set experiment.measure_cuda_sync=true \
  --set algorithm.params.n_steps=1024 --set algorithm.params.batch_size=128 \
  --set algorithm.params.frames_per_batch=256 --set evaluation.enabled=false \
  --output-dir outputs/hrl/gpu-learner
```

Ray interval pilot: all four trials use 4,096,000 LL decisions. The YAML controls
resources, concurrency, network and optimization overrides. Frozen configurations,
trial identities and source hashes are retained by the existing Tune integration.

```bash
# Configuration-only path; does not start Ray.
python scripts/ray_tune_trials.py --sweep-spec scripts/sweeps/hierarchical_intervals.yaml \
  --preflight-only
# Actual execution: run later on an allocated HPC Ray cluster.
python scripts/ray_tune_trials.py --sweep-spec scripts/sweeps/hierarchical_intervals.yaml \
  --root outputs/hrl/ray-intervals --trial-timeout 0
# Existing multi-node launcher for independent Tune trials (not shared multi-node HRL).
# Adjust the sweep YAML resources/concurrency before submission to use both nodes.
sbatch --nodes=2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=8 \
  --export=ALL,SWEEP_SPEC=scripts/sweeps/hierarchical_intervals.yaml \
  scripts/hpc/ray_tune.sbatch
```

## Files and reused components

Added: `src/hierarchical/{__init__,bank,selectors,task,policy,reporting}.py`,
`tests/test_hierarchical.py`, `configs/hierarchical/{bank,default,research}.yaml`,
`configs/maps/composite.yaml`, `configs/execution/hierarchical_cpu.yaml`,
`scenarios/ppo_{hierarchical,composite_scratch,composite_lora}.yaml`,
`scripts/sweeps/hierarchical_intervals.yaml`, and this report.

Modified: native TorchRL adapter; common PPO/checkpoint handling; TorchRL PPO and
updates; application train/evaluate; core configuration/provenance/task builder;
task-step contract; training algorithms/evaluation/collection/reporting; existing
Ray source inspection and sweep validation. Exact inventory:
`git diff c3446b8 --name-only`.

Reused: existing LoRA residual implementation and strict actor reconstruction;
RaceTask/action and reward composers; observation/track preview contracts;
TorchRL ClipPPOLoss, optimizers, minibatching and gradient clipping; existing
collectors, CPU/thread controls, asynchronous ranking and recovery; Hydra groups;
experiment launcher, Ray trial isolation and SLURM entry points. No runtime
dependency or alternate training/distributed framework was added.

## Final correction handoff: staged commands

All commands below are for the HPC after allocation and environment activation;
none were run as part of the final corrections. The historical 228-test result
above applies only to the earlier commit. The earlier snippets are retained as
concrete Stage C/D/G examples; use fresh directories for each run.

### A. Software, immutable inputs and regression gate

```bash
python --version
python -m pip check
sha256sum pretrained_models/circle_ppo_seed42_20k/best.pt pretrained_models/straight_best.pt pretrained_models/sweep_best.pt pretrained_models/hairpin_best.pt pretrained_models/chicane_best.pt pretrained_models/s_best.pt
cat configs/hierarchical/bank.yaml
python run.py --scenario scenarios/ppo_hierarchical.yaml --resolve-config
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
  tests/test_hierarchical.py tests/test_torchrl_ppo.py tests/test_torchrl_mappo.py \
  tests/test_ppo_lora.py tests/test_native_torchrl_env.py tests/test_checkpoints.py \
  tests/test_observation_contract.py tests/test_race_task.py tests/test_task_evaluation.py \
  tests/test_episode_reporting.py tests/test_on_policy_pipeline.py \
  tests/test_experiment_launcher.py tests/test_ray_tune_sweeps.py tests/test_ray_tune_trials.py
```

Compare all six SHA-256 values to the pinned bank manifest. The hierarchical
suite includes bank reconstruction/parity, categorical PPO, macro transitions,
duration-aware GAE, normalized dwell, return partition invariance, seeding and
preview contract checks. Validate checkpoint/resume and full suite results on HPC.

### B. Runtime correctness

In addition to the serial/process, asynchronous and resume commands above:

```bash
python run.py "${COMMON[@]}" "${SMOKE[@]}" --set experiment.parallel_envs=2 \
  --set experiment.envs_per_worker=2 --set evaluation.enabled=false \
  --output-dir outputs/hrl/smoke-multi-env
python run.py "${COMMON[@]}" --eval --checkpoint outputs/hrl/smoke-serial/final.pt \
  --eval-protocol selection --set hierarchical.record_trajectory=true \
  --output-dir outputs/hrl/smoke-learned-eval
python scripts/ray_tune_trials.py --sweep-spec scripts/sweeps/hierarchical_ray_smoke.yaml \
  --root outputs/hrl/ray-smoke --trial-timeout 0
```

### C and D. Specialization and switch compatibility

Run the six-policy, eight-validation-track matrix and 30 directed sequence
commands in the earlier blocks. Each policy uses the same seeds and spawn
schedule. Separate processes may be scheduled concurrently with unique output
directories; resource allocation is the operator's responsibility. The report
commands create per-map rows and skill/transition summaries; retain source JSON
for aggregate analysis. Approach trajectories are reproducible, not restored
from an identical simulator snapshot. No test map participates in selection.

### E. CPU scaling

Inspect available partitions and resources first:

```bash
sinfo -o '%P %a %l %D %c %m %G'
scontrol show partition
```

Use the preceding bounded scaling command as a template and choose workers from
`1 2 4 8 16 32`, envs per worker from `1 2 4 8`, thread count, `n_steps` and
`frames_per_batch` based on the allocation; do not run their Cartesian product.
Request CPUs for the learner, each collector, optional evaluation and Ray
head/worker overhead; actual topology and memory are cluster-specific. Compare
collection-only and end-to-end wall-time throughput separately, aggregate worker
CPU time separately from per-worker rates, and process RSS separately from the
bank tensor-storage bytes. The GPU learner command above is optional and later.

### F. Decision interval and preview experiments

The original `hierarchical_intervals.yaml` holds HL n_steps=1024, collection
batch=256, minibatch=128 and four epochs fixed. Increasing K increases LL
interactions per update; this is not LL-frequency control. The four
`hierarchical_ll_matched_k*.yaml` specifications couple `(K, n_steps, batch
frames)` as `(1,2560,640)`, `(5,512,128)`, `(10,256,64)`, `(20,128,32)`
with minibatch 128 and four epochs. They approximate 2560 LL decisions per
update, but vary HL samples and optimizer steps. Neither design controls all
optimization variables. Each has seed 42 and budget 4,096,000 LL decisions.
The latter are four independent one-trial Tune experiments rather than a
Cartesian product; modify resources and concurrency in the YAML as needed.

```bash
python scripts/ray_tune_trials.py --sweep-spec scripts/sweeps/hierarchical_intervals.yaml --root outputs/hrl/ray-fixed-hl --trial-timeout 0
for k in 1 5 10 20; do
  python scripts/ray_tune_trials.py --sweep-spec "scripts/sweeps/hierarchical_ll_matched_k${k}.yaml" --root "outputs/hrl/ray-ll-matched-k${k}" --trial-timeout 0
done
# After selecting K, edit the single interval value consistently in all three files.
for preview in ll 12m 18m; do
  python scripts/ray_tune_trials.py --sweep-spec "scripts/sweeps/hierarchical_preview_${preview}.yaml" --root "outputs/hrl/ray-preview-${preview}" --trial-timeout 0
done
```

For independent parallel trials across nodes use the existing `scripts/hpc/ray_tune.sbatch`
and shared filesystem as shown above. The launcher rejects dirty tracked source and
untracked files under `scripts`; put edited sweep copies under an output/config
directory outside those trees and reference the copies, or commit the edits first. Tune `resources_per_trial` and
`max_concurrent_trials` in the YAML before launch. Resume a failed experiment
with the same spec/root and `--restore`; retain frozen input configs, bank hashes,
run identities and each trial's native latest checkpoint. Each LL-matched and
preview file uses its own experiment root; concurrency between those invocations
must be arranged by the HPC scheduler. For a sweep W&B run, change its YAML
`wandb.mode` from `disabled` to `online` (or `offline`) and set `project`,
`entity` and a purpose/bank/sweep group there before launching. W&B receives
resolved config and trial identity, including K, preview, execution and seed;
choose group names for purpose and bank and use config filters for the rest.
For direct training omit `--no-wandb` and add `--wandb --set wandb.mode=online
--set wandb.group=<purpose-bank-sweep>`; use the existing project/entity settings.
No probability trajectories are sent to W&B.

### G. Final research experiments

Run multiple seeds for the chosen HL configuration, and the fixed, random,
geometry, pretrained base, direct PPO and composite LoRA baselines described
above. Rank the fixed adapter on composite validation only. Keep common seeds,
spawn schedules and LL budgets, and reserve held-out composite and external
maps until model and configuration selection are complete. No results or
performance claims are established by preparation alone.

## Continuing the hairpin adapter on the HPC

The saved `pretrained_models/hairpin_best.pt` is a model-only LoRA checkpoint.
A fresh run can initialize its full actor (including the trained adapter and
frozen base) and critic using `--checkpoint`; its optimizer, scheduler, RNG,
episode counters, and evaluation selection history start fresh. The checkpoint
provenance records both the original base identity and the adapter warm-start
file/hash. Matching network, observation, action, physics, and LoRA contracts are
required. `--resume` is instead recovery from a compatible `latest.pt`, retains
optimizer state, and requires the same original training budget; it cannot
extend a completed run's budget. Do not use `--resume` on `hairpin_best.pt`.

Choose an additional episode budget and an HPC execution configuration; this
example adds 60,000 episodes and keeps the one-lap family task and validation
split. It is an example budget, not an experimentally selected value. Run only
on an appropriate allocation, after the focused checks and configuration
resolution; this has not been executed locally.

```bash
python -m pytest -q tests/test_ppo_lora.py tests/test_checkpoints.py
python run.py --scenario scenarios/ppo_lora_families.yaml --set maps=generated_hairpin \
  --checkpoint pretrained_models/hairpin_best.pt --episodes 60000 \
  --no-wandb --no-render --output-dir outputs/hairpin-continued --resolve-config
python run.py --scenario scenarios/ppo_lora_families.yaml --set maps=generated_hairpin \
  --checkpoint pretrained_models/hairpin_best.pt --episodes 60000 \
  --no-wandb --no-render --output-dir outputs/hairpin-continued
# If interrupted, use the same scenario, map, budget, and output directory.
python run.py --scenario scenarios/ppo_lora_families.yaml --set maps=generated_hairpin \
  --episodes 60000 --no-wandb --no-render --resume outputs/hairpin-continued/latest.pt \
  --output-dir outputs/hairpin-continued
```

Configure workers, devices, W&B and CPU allocation for the machine before
launch; preserve the same settings on recovery where required. Compare the new
validation-selected checkpoint against the old one before choosing it. Keep the
original checkpoint and bank manifest unchanged during this run. A later bank
replacement requires an explicit update of the pinned path/hash and a new HL
run or compatible new provenance; previously trained HL checkpoints bind to the
old bank identity.

## Continue all five specialist adapters on the HPC

Each continuation starts from its own validation-selected LoRA checkpoint and
keeps the same PPO base, one-lap task, family training maps, and family
validation map. The five sources have the same pinned base SHA-256 and contain
rank-4 adapters. They are model-only: this starts five new optimizers, not the
original optimizer states. `base` is the pretrained PPO policy, not a specialist
adapter, and is not retrained by this procedure. No file in `pretrained_models/`
is overwritten or copied by these commands.

The existing independent-work-item launcher freezes one common source per root,
so use separate direct `run.py` processes for these five different sources.
These commands are **for an HPC allocation**, not this laptop. They are examples
using 60,000 additional episodes per family, one seed and serial execution;
choose the budget and resources for the allocation. The scenario defaults to a
CUDA learner and CPU collection. Adjust the learner device, worker count,
environments per worker, thread count, and output root before launching. Do not
oversubscribe the allocation; run individual family commands on independent
allocations when concurrent execution is desired. W&B remains disabled here.

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYGLET_HEADLESS=true
RUN_ROOT=outputs/adapter-continuation
EPISODES=60000
WORKERS=4
ENVS_PER_WORKER=1

declare -A adapter=(
  [straight]=pretrained_models/straight_best.pt
  [sweeper]=pretrained_models/sweep_best.pt
  [hairpin]=pretrained_models/hairpin_best.pt
  [chicane]=pretrained_models/chicane_best.pt
  [s_curve]=pretrained_models/s_best.pt
)

# Validate first; no training in the following two commands.
python -m pytest -q tests/test_ppo_lora.py tests/test_checkpoints.py
for family in straight sweeper hairpin chicane s_curve; do
  python run.py --scenario scenarios/ppo_lora_families.yaml \
    --set "maps=generated_${family}" --checkpoint "${adapter[$family]}" \
    --episodes "$EPISODES" --set experiment.parallel_envs="$WORKERS" \
    --set experiment.envs_per_worker="$ENVS_PER_WORKER" --torch-threads 1 \
    --no-wandb --no-render --output-dir "$RUN_ROOT/$family" --resolve-config
done

# Actual training: run only after selecting an allocation and budget.
for family in straight sweeper hairpin chicane s_curve; do
  python run.py --scenario scenarios/ppo_lora_families.yaml \
    --set "maps=generated_${family}" --checkpoint "${adapter[$family]}" \
    --episodes "$EPISODES" --set experiment.parallel_envs="$WORKERS" \
    --set experiment.envs_per_worker="$ENVS_PER_WORKER" --torch-threads 1 \
    --no-wandb --no-render --output-dir "$RUN_ROOT/$family" || break
done
```

For an interrupted family, retain its original budget, scenario, resources,
output path, and tracking settings; set `family` to that family and run:

```bash
family=hairpin
python run.py --scenario scenarios/ppo_lora_families.yaml \
  --set "maps=generated_${family}" --episodes "$EPISODES" \
  --set experiment.parallel_envs="$WORKERS" \
  --set experiment.envs_per_worker="$ENVS_PER_WORKER" --torch-threads 1 \
  --no-wandb --no-render --resume "$RUN_ROOT/$family/latest.pt" \
  --output-dir "$RUN_ROOT/$family"
```

For every family, compare the original checkpoint with the newly selected
`best.pt` on the **same family validation protocol**, seeds, spawns and metrics.
These evaluations are independent and may be scheduled concurrently on the
HPC with sufficient resources. Do not use held-out test tracks to select a
checkpoint.

```bash
for family in straight sweeper hairpin chicane s_curve; do
  for version in original continued; do
    checkpoint="${adapter[$family]}"
    if [ "$version" = continued ]; then checkpoint="$RUN_ROOT/$family/best.pt"; fi
    python run.py --scenario scenarios/ppo_lora_families.yaml \
      --set "maps=generated_${family}" --eval --eval-protocol selection \
      --checkpoint "$checkpoint" --no-wandb --no-render \
      --output-dir "$RUN_ROOT/comparison/$family/$version" || break 2
  done
done
```

Inspect `evaluation_history.jsonl` over training and the paired
`evaluation_report.json` outputs for completion, collisions, progress and clean
lap time. A long flat validation curve is evidence to investigate, not proof of
convergence: eight evaluations on one validation map are limited, and a best
checkpoint can be selected due to noise. Repeat continuations with distinct
training seeds and fresh output roots when resources permit, keeping paired
validation seeds fixed; consider additional validation maps before selecting a
final bank. Preserve all source hashes and selected outputs. Update the bank
manifest only after selection; existing HL checkpoints bind to the old hashes.
