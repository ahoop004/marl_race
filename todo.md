# MARL Race — Pre-Profiling Refactor Tasks

**Repository:** `ahoop004/marl_race`  
**Branch:** `main`  
**Scope:** Serial, single-environment PPO/MAPPO using TorchRL

**Completion status:** Tasks 1–13 complete.

- Task 11: Optional exclusive stage timing, initialization/warm-up separation, and environment/physics/learner/optimizer throughput are recorded through existing metrics. Enable detailed timing with `--set execution.experiment.measure_stages=true` on a serial profile. The first update is warm-up by default and counts toward the training budget.
- Task 12: Startup provenance records runtime versions, devices, threads, seeds, model contracts, configuration hashes, and map image/centerline/wall hashes. Older checkpoints retain their existing YAML recovery checks; new checkpoints also verify recorded geometry.
- Task 13: Targeted validation passed: 202 tests, 2 CUDA checks skipped because CUDA is unavailable. Final timing follow-up: 30 tests passed. Nine-decision PPO/MAPPO smoke runs verified finite losses, actor/critic updates, checkpoint artifacts, counters, stage timings, and metadata. Smoke-only overrides used smaller networks and rollouts; profiling defaults remain unchanged.

**Remaining constraints:** Detailed CUDA timing synchronizes stage boundaries and is disabled by default. Warm-up spans the configured first update(s), so steady-state statistics require a longer budget. Production PPO transfer from the current pretraining checkpoint requires `--max-speed 8` to match its strict wheel-speed contract. No full training experiments or expensive benchmarks were run.

## Agent guidelines

- Reuse existing modules, classes, and configuration mechanisms wherever possible.
- Do not introduce parallel environments, multiprocessing, Ray, MPI, or distributed training.
- Do not introduce new RL algorithms or replace TorchRL components.
- Avoid generating additional documentation or unnecessary abstraction layers.
- Do not create extensive new test suites. Update existing tests where necessary.
- Do not run full training experiments or expensive benchmarks.
- Preserve the existing experimental scenarios and checkpoint compatibility wherever possible.
- Make focused changes and report affected files, design decisions, and any remaining issues.
- Distinguish profiling-specific settings from production training hyperparameters.

---

## P0 — Required before profiling

### Task 1. Standardize rollout collection and update semantics

**Files:**
- `src/training/on_policy.py`
- `src/agents/torchrl_ppo.py`
- `src/agents/torchrl_mappo.py`

**Actions:**
1. Review the existing collector interruption conditions and update triggers.
2. Make `n_steps` a clearly defined learning-sample accumulation target, distinct from the collector's physical batch size.
3. Allow MAPPO to accumulate experience across completed episodes rather than automatically optimizing at every episode boundary.
4. Preserve episode boundaries when calculating GAE and bootstrapped returns.
5. Ensure inactive or retired agents do not contribute invalid actor training samples.
6. Preserve valid joint-team credit assignment for remaining learners.
7. Support partial final rollouts when a training budget is exhausted.
8. Verify that padded collector rows cannot enter optimization.
9. Keep the implementation compatible with TorchRL's supported collector interfaces.

**Acceptance criteria:** PPO and MAPPO accumulate correctly bounded rollout data, preserve terminal/truncation semantics, and have predictable optimizer-update behavior independent of episode length.

### Task 2. Introduce lightweight serial profiling presets

**Files:**
- `configs/algorithm/ppo.yaml`
- `configs/algorithm/mappo.yaml`
- `configs/execution/`
- Relevant scenario configurations

**Actions:**
1. Create minimal Hydra profiling presets or overrides that reuse existing configuration groups.
2. Use the following initial settings:

| Parameter | PPO | MAPPO |
|---|---:|---:|
| `n_steps` | 4096 | 2048 |
| `batch_size` | 512 | 512 |
| `n_epochs` | 4 | 4 |
| `learning_rate` | 0.001 | 0.0001 |
| `lr_schedule` | constant | constant |
| `gamma` | 0.99 | 0.99 |
| `gae_lambda` | 0.95 | 0.95 |
| `clip_range` | 0.2 | 0.2 |
| `ent_coef` | 0.01 | 0.01 |
| `vf_coef` | 0.5 | 0.5 |
| `max_grad_norm` | 0.5 | 0.5 |
| `device` | cpu | cpu |
| `torch_threads` | 1 | 1 |

3. Ensure the PPO constant learning-rate configuration does not retain a conflicting `learning_rate_end`.
4. Keep the production training parameters separate from the profiling presets.
5. Support configuring the step budget through the existing CLI/Hydra mechanisms.

**Acceptance criteria:** Profiling runs can be configured reproducibly without modifying the production experiment definitions.

### Task 3. Verify training budgets and counters

**Files:**
- `src/training/on_policy.py`
- `src/training/race_reporting.py`
- `src/training/hooks.py`

**Actions:**
1. Verify that `total_steps` counts joint environment decisions, including decisions after learner retirement when fixed opponents remain active.
2. Keep separate counters for:
   - Environment decisions
   - Physics integration steps
   - Active learner samples
   - Completed episodes
   - Policy updates
   - Optimizer steps
3. Record actual learner samples and environment decisions per update.
4. Ensure budget termination does not introduce artificial environment transitions or invalid samples.
5. Remove ambiguity between `episodes` and `total_steps` in profiling configurations.
6. Verify that reported steps remain consistent across console, CSV, and W&B logging.

**Acceptance criteria:** Step counters agree with observed simulation and optimizer activity, including partial episodes and terminated learners.

### Task 4. Standardize CPU/CUDA configuration

**Files:**
- `configs/execution/`
- `src/utils/torch_io.py`
- `src/training/on_policy.py`

**Actions:**
1. Use CPU for the default serial profiling configuration.
2. Allow explicit CUDA selection through existing overrides.
3. Verify correct device handling for:
   - Environment tensors
   - Policy inference
   - Rollout storage
   - GAE
   - Optimizer updates
4. Avoid unnecessary device transfers or conversions.
5. Confirm the selected device is clearly recorded in run metadata.
6. Preserve correct synchronization when measuring CUDA execution.

**Acceptance criteria:** The same profiling scenario runs on CPU or CUDA without changing task semantics.

### Task 5. Fix avoidable GPU synchronization overhead

**File:**
- `src/agents/torchrl_updates.py`

**Actions:**
1. Inspect per-minibatch tensor-to-Python conversions, especially KL calculations.
2. Avoid unnecessary CUDA synchronization when `target_kl` is disabled.
3. Preserve KL early stopping when explicitly enabled.
4. Avoid excessive per-minibatch CPU transfers for logging.
5. Keep the loss functions, gradient calculations, and optimization behavior unchanged.

**Acceptance criteria:** Unnecessary synchronization is reduced without changing computed losses or optimizer behavior.

---

## P1 — Configuration and scientific consistency

### Task 6. Consolidate completion reward configurations

**Files:**
- `configs/reward/solo_completion.yaml`
- `configs/reward/solo_transfer.yaml`
- `configs/reward/team_completion.yaml`
- `scenarios/ppo_lap_completion_transfer*.yaml`
- `src/wrappers/rewards/completion.py`

**Actions:**
1. Use normalized signed lap progress as the standard completion reward.
2. Align the individual reward components between PPO pretraining and MAPPO completion.
3. Preserve the additional MAPPO both-teammates-finish bonus.
4. Verify fixed-denominator mean team rewards after one learner retires.
5. Confirm collision penalties are applied appropriately, including when collision does not terminate the episode.
6. Verify that boundary events, clean completion, and timeouts are handled consistently.
7. Retain the previous metre-based transfer reward only as an explicitly named alternative if needed.

**Acceptance criteria:** Solo and team completion use equivalent individual reward definitions, while MAPPO retains its cooperative objective. Existing alternative experiments remain identifiable.

### Task 7. Align episode termination and evaluation protocols

**Files:**
- `configs/scenario/`
- `configs/evaluation/`
- `scenarios/`
- `src/training/evaluation_config.py`

**Actions:**
1. Verify training and evaluation agreement on:
   - Target laps
   - Episode termination conditions
   - Collision termination
   - Track-limit termination
   - Maximum episode length
2. Review the five-lap versus twenty-lap mismatch in the PPO transfer scenario.
3. Ensure profiling scenarios have finite episode limits.
4. Preserve individual learner retirement in MAPPO without prematurely ending teammates' episodes.
5. Verify correct distinction between termination and truncation.
6. Preserve fixed-opponent behavior and evaluation seeds.

**Acceptance criteria:** Episode and evaluation horizons are explicit, consistent, and reproducible.

### Task 8. Standardize checkpoint source resolution

**Files:**
- `configs/adaptation/`
- `src/application/checkpoints.py`
- `src/application/train.py`
- `src/agents/common/checkpoints.py`

**Actions:**
1. Review the hard-coded `outputs/pretrain/best.pt` checkpoint source.
2. Ensure pretrained model paths resolve correctly from configurable locations.
3. Support sharing one exact PPO source checkpoint across full fine-tuning and LoRA runs.
4. Preserve strict actor, critic, and observation compatibility checks.
5. Verify that MAPPO transfer initializes a fresh centralized critic.
6. Verify that LoRA freezes the pretrained actor base and trains only intended actor parameters.
7. Preserve distinct semantics for `best.pt`, `latest.pt`, and `final.pt`.
8. Retain the documented fresh-episode restart behavior on resume.

**Acceptance criteria:** Transfer experiments use the intended checkpoint and save independently recoverable model states.

### Task 9. Standardize simulation parameters for profiling

**Files:**
- `configs/simulation/default.yaml`
- `configs/vehicle/combined_slip.yaml`
- `configs/scenario/`

**Actions:**
1. Establish one controlled simulation setup for serial profiling.
2. Use the following initial settings:

| Parameter | Value |
|---|---|
| Timestep | 0.05 s |
| Action repeat | 1 |
| Max speed | 8 m/s |
| LiDAR beams | 108 |
| LiDAR range | 10 m |
| Map | `circle_map` |
| Friction | Fixed, μ = 1.0489 |
| Seed | 42 |
| Rendering | Disabled |

3. Preserve the nonlinear combined-slip vehicle model.
4. Preserve existing wheel actuator and steering constraints.
5. Verify that the configured speed limit applies consistently to learners and MPC opponents.
6. Keep the numerical integration settings unchanged.
7. Ensure deterministic map, spawn, and friction initialization under fixed seeds.

**Acceptance criteria:** Repeated serial profiling runs use equivalent simulation settings. Randomized friction remains available for production training.

### Task 10. Separate expensive experiment services from profiling

**Files:**
- `src/application/train.py`
- `src/training/hooks.py`
- `configs/evaluation/`
- `configs/wandb.yaml`

**Actions:**
1. Disable periodic checkpoint-selection evaluations in profiling presets.
2. Disable W&B and rendering during baseline measurements.
3. Retain normal lightweight console/CSV reporting.
4. Avoid per-transition disk writes unless explicitly requested.
5. Keep end-of-run checkpoint functionality.
6. Make evaluation, checkpoint, and logging overhead measurable separately.
7. Avoid modifying the behavior of normal production experiments.

**Acceptance criteria:** Core training throughput can be measured without expensive optional services, while their overhead can be measured independently later.

---

## P1 — Profiling preparation

### Task 11. Prepare stage-level performance instrumentation

**Files:**
- `src/training/on_policy.py`
- `src/training/race_reporting.py`
- `src/agents/torchrl_updates.py`
- Existing environment/physics modules where appropriate

**Actions:**
1. Reuse existing performance metrics and hooks.
2. Add lightweight, opt-in timing where needed.
3. Support separate measurement of:
   - Environment stepping
   - Physics integration
   - LiDAR
   - Observation/reward processing
   - Policy inference
   - Collector overhead
   - GAE
   - PPO/MAPPO optimization
   - Tensor/device transfers
4. Distinguish initialization and JIT warm-up from steady-state measurements.
5. Avoid excessive timing calls inside high-frequency simulation loops when detailed instrumentation is disabled.
6. Record collection, update, evaluation, and checkpoint times separately.
7. Report environment decisions/sec, physics steps/sec, learner samples/sec, and optimizer steps/sec.

**Acceptance criteria:** Profiling can identify whether simulation, inference, collection, or optimization dominates wall-clock time, without requiring a new logging framework.

### Task 12. Improve profiling reproducibility metadata

**Files:**
- `src/core/provenance.py`
- Existing logging/checkpoint utilities

**Actions:**
1. Verify that resolved configurations and their hashes are retained.
2. Record:
   - Git commit and dirty state
   - Python version
   - PyTorch/TorchRL/TensorDict versions
   - Device and CUDA runtime information
   - CPU thread configuration
   - Random seeds
   - Physics and observation contracts
3. Extend map provenance to include relevant map geometry and centerline files, not just map YAML.
4. Preserve existing checkpoint provenance and avoid duplicating metadata.
5. Keep metadata collection inexpensive and performed at startup rather than per step.

**Acceptance criteria:** Profiling outputs identify their software, hardware, and experimental configuration sufficiently for repeatable comparisons.

---

## P2 — Final validation

### Task 13. Perform focused functional validation

**Files:**
- Existing tests under `tests/`

**Actions:**
1. Update existing tests affected by the collection-cadence changes.
2. Run only the relevant targeted tests covering:
   - PPO collection and optimization
   - MAPPO collection and optimization
   - Episode boundaries and GAE
   - Retired agent masking
   - Hydra configuration
   - Checkpoint transfer/resume
3. Perform one short PPO and one short MAPPO end-to-end smoke run.
4. Verify finite losses and that intended parameters receive updates.
5. Confirm the expected checkpoint artifacts and counters are created.
6. Do not run full training, large hyperparameter sweeps, or expensive evaluation batches.

**Acceptance criteria:** Core serial training functionality passes targeted checks and is ready for performance measurement.

---

## Deferred — Do not implement in this refactor

- Multi-environment collection
- Distributed/multinode training
- Ray or MPI integration
- Major neural network architecture changes
- Hyperparameter optimization
- Gamma/GAE tuning experiments
- LoRA rank or adapter architecture experiments
- New racing tasks or opponent algorithms
- Extensive documentation generation
- Full-scale training comparisons
- Broad test-suite expansion

These can be addressed after the serial baseline is established.

## Completion deliverables

When finished, provide:

1. A concise summary of changes by task.
2. Files modified.
3. Final effective serial profiling parameters.
4. Results of targeted validation.
5. Known issues or remaining assumptions.
6. Any changes that could affect checkpoint compatibility or experiment comparability.

**Do not launch full profiling or training experiments. The goal is to leave the repository ready for the next profiling phase.**
