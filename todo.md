Implement **Grouped Synchronous TorchRL Environment Collection** in `ahoop004/marl_race`, branch `parallel`.

### Objective

Extend the existing TorchRL synchronous PPO/MAPPO collector so each CPU process can manage multiple independent racing environments.

Keep the current TorchRL collector, PPO/MAPPO learners, TensorDict data format, existing checkpoint format, and training algorithms.

Do not recreate the custom multiprocessing scheduler from `f110_marl`.

The goal is to compare:

- Current: 8 processes × 1 environment = 8 races
- Grouped: 8 processes × 2 environments = 16 races
- Grouped: 8 processes × 4 environments = 32 races
- Grouped: 8 processes × 8 environments = 64 races

These are independent race environments, not vehicles within the same race. Each MAPPO environment must retain its complete 2v2 simulation.

### Background

The old `F110_MARL` repository on `marl-rewiring` supported separate environment and worker counts, including 400 environments across 100 processes.

It used grouped environments, custom multiprocessing, centralized batched inference, and a synchronous PPO/MAPPO learner.

The new `marl_race` project uses TorchRL 0.14 with one race per process.

Recent HPC PPO measurements on 32 allocated CPUs and one NVIDIA L40S showed:

| Workers | Overall decisions/s | Steady decisions/s |
|---|---|---|
| 8 | 472.12 | 509.21 |
| 16 | 372.49 | 396.64 |
| 28 | 272.99 | 295.40 |

Steady collection time increased from 91.60 to 166.44 seconds between 8 and 28 workers. Aggregation increased from 19.39 to 28.21 seconds. Weight synchronization remained below 0.5 seconds.

The current physical collection batch is 256 frames, which gives individual workers increasingly short collection fragments as worker count increases.

We want to evaluate grouped environments rather than continue increasing process count.

### Task 1 — Add environment grouping configuration

Add one optional parameter:

```yaml
experiment:
  parallel_envs: 8
  envs_per_worker: 4
```

Maintain backward compatibility:

- `parallel_envs` remains the number of collector processes.
- `envs_per_worker` is the number of independent race environments within each process.
- Default `envs_per_worker=1`.
- Total environments = `parallel_envs * envs_per_worker`.
- `1 × 1` must retain the existing direct serial path.
- Grouped configurations must use the appropriate process collector path, including when there is only one process.
- Validate both as positive integers.

Do not introduce additional configuration groups, new scenario files, or unrelated parameter changes.

### Task 2 — Use TorchRL SerialEnv inside existing process collectors

Investigate and use the existing TorchRL 0.14 `SerialEnv` API.

Reuse:

- `Collector(backend="process", sync=True)`
- `NativeRaceTorchRLEnv`
- `RaceEnvFactory`
- Existing PPO/MAPPO collection policies
- Existing learner-side weight synchronization

Each process should construct a `SerialEnv` containing its assigned independent racing environments.

Each nested environment must have:

- Its own race task and simulator instance
- Independent controller state
- Independent seed and map schedule
- Independent episode/reset lifecycle
- Unique global environment ID

Use global environment IDs to derive seeds, for example:

`environment_id = worker_id * envs_per_worker + local_env_id`

Preserve the current deterministic seed convention using the global environment ID.

Do not introduce nested `ParallelEnv` process pools, Ray, MPI, DDP, or custom multiprocessing communication.

Keep inference on CPU inside each collector process for the initial implementation.

Allow the central learner to remain on CPU or CUDA.

### Task 3 — Handle grouped TensorDict batch dimensions

The current process collector assumes one environment per worker.

With grouped environments, TorchRL should preserve dimensions resembling:

`[workers, envs_per_worker, time]`

Confirm the actual dimensions returned by the installed TorchRL version.

Update existing batch aggregation and rollout assembly to support both ungrouped and grouped layouts.

Requirements:

1. Preserve each environment's chronological trajectory order.
2. Never concatenate unrelated environments into a single GAE trajectory.
3. Split trajectories correctly at physical episode boundaries.
4. Preserve bootstrap and termination semantics.
5. Preserve active learner masks and retired-agent behavior.
6. Preserve MAPPO team rewards and shared-critic inputs.
7. Exclude fixed-only continuation decisions from learner samples while still counting physical environment decisions.
8. Preserve policy-version checks to prevent stale-policy updates.
9. Preserve the current synchronous update barrier.

Do not modify PPO or MAPPO losses, networks, optimizer equations, or GAE implementations.

Do not flatten environment and agent dimensions interchangeably.

### Task 4 — Make reporting compatible with grouped environments

The current process collector contains process-local reporting state designed around one environment per process.

Adapt this with minimal changes so multiple environments can report independently from the same process.

Ensure:

- Unique global environment IDs
- Unique episode identities
- Correct per-environment episode counters
- Correct process-level counters
- Correct total environment decisions
- Correct active learner sample counts
- Correct MAPPO team outcomes
- No duplicate or lost episode records
- Correct policy versions and worker attribution

Investigate compatibility of the existing `NonTensorData` race-event transport with `SerialEnv`.

Use the existing event/reporting infrastructure whenever possible.

Do not build a new logging system.

### Task 5 — Preserve training budget semantics

Maintain the existing contracts:

- `experiment.total_steps`: Global joint environment decisions across all environments.
- `algorithm.params.n_steps`: Global active learner sample target per update.
- `algorithm.params.frames_per_batch`: Global physical collection batch target.
- `algorithm.params.batch_size`: Optimizer minibatch size.
- `algorithm.params.n_epochs`: PPO/MAPPO optimizer epochs.

Increasing the number of environments must not automatically multiply the global learner sample target or optimizer workload.

Handle collector batch rounding and sample overshoot explicitly and report them using existing metrics.

Do not introduce artificial episode truncations merely to make batch sizes divisible.

Preserve the current behavior when grouping is disabled.

### Task 6 — Minimal profiling additions

Reuse the existing timers and run summaries.

Add only the missing grouping information:

- Number of processes
- Environments per process
- Total active environment slots
- Actual collected decisions
- Active learner samples
- Collection wall time
- Aggregation time
- Weight synchronization time
- Learner update time
- Per-process collection time
- Collector startup time

Retain current worker-waiting metrics.

Make clear that aggregate worker-wait time overlaps collection wall time and must not be added to it.

Do not add per-step logging, a new profiling framework, or large diagnostic outputs.

Preserve normal training behavior when detailed profiling is disabled.

### Task 7 — Minimal correctness validation

Perform short, targeted smoke tests only.

Required configurations:

**PPO**

- 1 process × 1 environment: existing behavior
- 2 processes × 2 environments: grouped collection

**MAPPO**

- 1 process × 1 environment: existing behavior
- 2 processes × 2 environments: grouped collection

For grouped MAPPO, use the actual hybrid PP/FTG completion scenario, not the MPC serial profiling scenario.

Check:

- Correct TensorDict batch dimensions
- Correct environment IDs and seeds
- Correct collected-step accounting
- Correct episode and GAE boundaries
- Valid PPO/MAPPO optimization
- No stale policy versions
- Clean worker shutdown
- CPU collection with optional CUDA learner

Use small decision budgets.

Do not add an extensive test suite or run long training jobs.

If TorchRL `SerialEnv` has an incompatibility with the existing dynamic metadata or event transport, identify the precise blocker and attempt only a small, localized fix.

If the fix requires replacing the collector or rewriting the training pipeline, stop and report the blocker instead.

### Task 8 — Controlled performance comparison

After correctness validation, perform only a small local comparison.

Use the same:

- PPO scenario
- Git commit
- Seed
- Physics and reward configuration
- Global environment decision budget
- Global PPO sample target per update
- Minibatch size
- Epoch count
- Learner device

Compare:

1. 2 processes × 1 environment
2. 2 processes × 2 environments
3. 2 processes × 4 environments, if the earlier configurations pass

Use CPU inference and a single learner.

Keep detailed stage profiling, controller profiling, evaluation, and W&B disabled.

Run only enough decisions to establish that grouped collection works and to identify obvious regressions.

Do not run the full HPC scaling matrix. Provide commands for those experiments so they can be executed separately on the HPC node.

For later HPC comparisons, support two controlled benchmark types:

**A. Fixed global collection batch**

Keep `frames_per_batch` constant while changing grouping. This isolates the effect of environment grouping and changes fragment length per environment.

**B. Fixed approximate fragment length**

Increase `frames_per_batch` with total environment count to maintain approximately the same decisions per environment per collector batch. Report changes in collector batch granularity and sample overshoot.

Do not mix these comparisons when claiming performance improvements.

### Strict implementation constraints

- Remain on TorchRL 0.14.
- Reuse existing training and collection infrastructure.
- Minimal additional code.
- Prefer modifying existing files rather than adding new modules.
- No custom collector framework.
- No changes to PPO/MAPPO algorithms or rewards.
- No new neural network architectures.
- No distributed/multi-node features.
- No centralized inference server yet.
- No Ray, MPI, or DDP.
- No README updates.
- No documentation files.
- No unnecessary test files.
- No full regression suite.
- No extensive profiling or long training runs.
- Do not automatically change Python or package versions.

### Deliverables

After implementation, provide a concise report containing:

1. Files changed and why.
2. How `envs_per_worker` is configured.
3. Actual grouped TensorDict batch layout.
4. How per-environment reporting and GAE boundaries are preserved.
5. Smoke-test results.
6. Small local throughput comparison.
7. Exact HPC benchmark commands for 8×1, 8×2, 8×4, and 8×8.
8. Any known limitations or performance regressions.

Do not proceed to centralized batched inference or a custom collector without approval.