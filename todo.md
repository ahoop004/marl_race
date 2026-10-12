# Task: Final TorchRL Collection and Training Pipeline Optimization

**Repository:** `ahoop004/marl_race`  
**Branch:** `parallel`  
**Objective:** Optimize PPO/MAPPO execution before the final HPC performance evaluation and full training experiments.

## Context

The project has completed its initial refactoring, serial profiling, synchronous TorchRL multiprocessing implementation, and grouped-environment implementation.

The current architecture uses:

- TorchRL 0.14 synchronous `Collector`
- CPU collection with a CPU or CUDA learner
- Configurable `experiment.parallel_envs`
- Configurable `experiment.envs_per_worker`
- Grouped `SerialEnv` instances within collector processes
- TorchRL/TensorDict rollout storage
- On-policy policy-version synchronization
- PPO and MAPPO learners, including existing transfer and LoRA support

The current best HPC PPO configuration is:

- 8 collector processes
- 2 racing environments per process
- 16 total racing environments
- `frames_per_batch=1024`
- `n_steps=4096`
- `batch_size=512`
- `n_epochs=4`
- CPU inference
- CUDA learner
- 32 allocated CPUs
- NVIDIA L40S GPU

Measured performance:

- Overall throughput: **684.45 environment decisions/sec**
- Steady-state throughput: **765.78 decisions/sec**
- Collection: 54.70 seconds
- Aggregation: 16.30 seconds
- Update: 8.98 seconds
- Total decisions: 65,536
- PPO updates: 16

Earlier benchmarks demonstrated that both increasing collection batch size and grouping environments improve throughput.

We now want to optimize the remaining overhead without destabilizing the existing training implementation.

## General instructions

Prioritize code reuse and minimal implementation changes.

Preserve:

- Existing TorchRL 0.14 infrastructure
- PPO/MAPPO mathematical behavior
- Actor and critic architectures
- LoRA adapters and transfer learning
- Reward functions and episode semantics
- Observation/action contracts
- Checkpoint formats and resume behavior
- Existing experiment configuration compatibility
- Deterministic environment IDs and seed derivation
- Synchronous on-policy collection and policy-version correctness

Do not:

- Replace TorchRL
- Reimplement PPO/MAPPO
- Introduce Ray, MPI, DDP, or multi-node execution
- Change package versions
- Change learning hyperparameters as an optimization
- Add broad abstractions or frameworks
- Generate README files or additional documentation
- Introduce a large test suite
- Run expensive HPC experiments locally
- Rewrite working code without demonstrated performance benefit

Keep the existing ungrouped and grouped collectors functional throughout.

Use focused verification and incremental performance measurements.

---

# Phase 1 — Optimize TensorDict Aggregation

**Priority: Critical**

### Relevant code

- `src/training/on_policy.py`
- `_receive_process_batch()`
- `_process_streams()`
- `_process_rollouts()`
- Collection and aggregation paths within `train()`

### Problem

The current aggregation implementation iterates over environments and individual transitions in Python.

Each transition may require:

- Extracting scalar counters from tensors
- Checking episode IDs
- Processing agent masks
- Inspecting reporting events
- Updating counters
- Tracking episode/policy information

With larger grouped environment counts, this introduces significant Python overhead.

### Required changes

1. Identify which aggregation operations dominate runtime.
2. Replace per-transition tensor operations with vectorized operations wherever correctness permits.
3. Aggregate environment decisions, physics steps, and learner sample counts using tensor reductions.
4. Preserve independent trajectories for each environment.
5. Optimize episode-boundary detection using tensor operations.
6. Minimize repeated `.item()`, `.unbind()`, tensor indexing, and conversion operations.
7. Reduce avoidable TensorDict cloning, concatenation, and intermediate allocations.
8. Process variable-length Python reporting events only where necessary.

Do not flatten environment, time, and agent dimensions incorrectly.

Do not merge GAE sequences across physical episode boundaries.

### Important correctness requirements

Preserve:

- Active learner masks
- Fixed-only continuation decisions
- Terminal rewards
- Truncation versus termination
- Bootstrap values
- Team GAE behavior
- Policy-version validation
- Exact environment and learner sample accounting

Avoid introducing tensor views that become invalid when collector buffers are reused.

### Acceptance criteria

Aggregation produces equivalent sample counts, episode boundaries, and learning inputs to the original implementation.

Report before/after aggregation time and throughput for a short matched comparison.

---

# Phase 2 — Reduce Worker Reporting Overhead

**Priority: High**

### Relevant code

- `src/training/process_collection.py`
- `src/training/race_reporting.py`
- `src/adapters/native_torchrl.py`
- Existing training hooks and metric reporting

### Problem

Every environment transition can carry reporting metadata, event payloads, and Python objects through the TensorDict collection pipeline.

Grouped collection increases the number of environments contributing reports to each batch.

### Required changes

1. Examine the current `NonTensorData` event transport.
2. Identify allocations and serialization operations performed on every environment step.
3. Avoid transmitting empty reporting data unnecessarily where TorchRL's tensor/spec contracts permit.
4. Batch or buffer worker events when safe.
5. Avoid repeatedly creating identical metadata.
6. Aggregate routine counters within worker processes where possible.
7. Preserve detailed reporting only for the events or metrics that require it.
8. Maintain compatibility with existing CSV and W&B hooks.

Investigate whether normal training can use a minimal reporting path without losing required metrics.

Do not disable episode logging, racing outcomes, rewards, or required performance counters merely to improve throughput.

### Requirements

The following must remain correct:

- Episode starts and ends
- Collision and completion events
- Individual and team race outcomes
- Environment IDs
- Episode IDs
- Reward totals
- Training progress
- Evaluation and checkpoint triggers
- Per-worker profiling data

Preserve the existing reporting schema unless changing an internal field is unavoidable.

### Acceptance criteria

Reduced reporting overhead with no missing, duplicated, or reordered required lifecycle events.

Measure the performance effect independently from Phase 1 wherever practical.

---

# Phase 3 — Optimize Environment Tensor Construction

**Priority: High**

### Relevant code

- `src/adapters/native_torchrl.py`
- `NativeRaceTorchRLEnv._tensor()`
- `NativeRaceTorchRLEnv._output()`
- Environment action conversion
- TensorDict construction

### Problem

The environment creates many small tensors using `torch.tensor()` on each observation and transition.

Repeated memory allocation, copying, spec validation, and TensorDict construction may be costly.

### Required changes

Investigate and selectively optimize:

1. Repeated creation of small CPU tensors.
2. Conversions from NumPy arrays to PyTorch tensors.
3. Repeated creation of constant index/mask tensors.
4. TensorDict construction overhead.
5. Redundant copies or dtype conversions.
6. Runtime spec/type checking overhead.
7. Repeated action conversion from tensors to NumPy arrays.

Consider:

- Reusing immutable tensors
- Using `torch.from_numpy()` or `torch.as_tensor()` when safe
- Preallocating reusable temporary buffers
- Constructing outputs from fewer allocations
- Disabling expensive runtime validation after initialization, if safe

### Critical safety requirement

The collector must retain immutable snapshots of transitions until they are consumed.

Do not return tensor views backed by mutable simulation state if later environment steps can overwrite them.

Verify tensor storage lifetime, especially with grouped `SerialEnv` and `use_buffers=False`.

The optimized implementation must preserve:

- Tensor shapes and dtypes
- Device placement
- Observation ordering
- Agent masks
- Reward values
- Termination/truncation semantics
- TorchRL environment specifications

Runtime type-check disabling must be separately measurable and reversible.

### Acceptance criteria

Equivalent environment outputs and no transition aliasing or storage corruption.

Record the impact on collection throughput and per-step allocation/conversion time.

---

# Phase 4 — Collection Batch Configuration

**Priority: Benchmark tuning, not architectural rewriting**

The existing implementation already supports configurable `frames_per_batch`.

Do not introduce a new batching framework.

Verify that collection parameters remain independent:

- `parallel_envs`
- `envs_per_worker`
- `frames_per_batch`
- `n_steps`
- `batch_size`
- `n_epochs`

Confirm that increasing collection frames does not implicitly change:

- PPO/MAPPO target learner sample budget
- Optimizer epochs
- Optimizer minibatch size
- Environment termination behavior

Continue reporting actual collected frames and overshoot.

Avoid introducing rollout truncations solely to produce divisible batches.

Prepare benchmark overrides for:

- `frames_per_batch=512`
- `frames_per_batch=1024`
- `frames_per_batch=2048`

Do not change default training hyperparameters before the final evaluation.

---

# Phase 5 — CPU Affinity and NUMA Awareness

**Priority: Experimental**

The target HPC node contains:

- 192 CPU cores
- Two CPU sockets
- One or more NVIDIA L40S GPUs
- SLURM-managed CPU allocations

The initial benchmarking allocation uses 32 CPU cores and one GPU.

### Required investigation

1. Check how TorchRL collector processes inherit CPU affinity.
2. Verify that workers remain inside their SLURM allocation.
3. Determine whether collector workers are concentrated on one socket.
4. Investigate whether process placement contributes to synchronization or collection overhead.
5. Compare current OS scheduling against explicitly controlled process placement.

Prefer SLURM CPU binding or existing operating-system affinity mechanisms rather than a custom scheduling implementation.

If code-level affinity control is necessary:

- Make it optional.
- Respect `os.sched_getaffinity()`.
- Never assign CPUs outside the allocated CPU set.
- Avoid oversubscribing learner and collector processes.
- Preserve default behavior unless affinity is explicitly enabled.

Do not hardcode hardware-specific CPU IDs.

Report whether affinity improves or reduces throughput.

---

# Phase 6 — Prototype Centralized TorchRL Inference

**Priority: Conditional architecture experiment**

Only begin this phase after Phases 1–3 are implemented, measured, and stable.

Do not replace the existing grouped process collector.

### Goal

Compare the current worker-local inference architecture against centralized batched inference, similar to the older `F110_MARL` implementation.

The older implementation used multiple environments per CPU process and performed batched actor/critic inference in the parent process.

### Preferred TorchRL prototype

Investigate using:

- TorchRL `ParallelEnv`
- Nested grouped `SerialEnv`
- `Collector(backend="direct")`
- Central batched TorchRL policy execution

Conceptual layout:

CPU workers → grouped racing environments → central batched actor → actions returned to workers → synchronous PPO/MAPPO learner.

### Implementation requirements

1. Reuse existing `RaceEnvFactory`.
2. Reuse existing native TorchRL environment adapter.
3. Reuse current TorchRL actors, critics, and learners.
4. Preserve the current grouped process collector as the default.
5. Introduce only one optional backend selection if necessary.
6. Support centralized CPU inference initially.
7. Support centralized CUDA inference if the API and policy device contracts permit.
8. Preserve exact on-policy update barriers and policy versions.

Verify that `ParallelEnv`, grouped environments, and the existing event transport are compatible with the installed TorchRL version.

Check action/observation batch shapes, policy-device transfers, and subprocess lifecycle before attempting larger benchmarks.

### Important restrictions

Do not introduce an asynchronous collector into normal PPO/MAPPO training.

Do not implement a custom inference server unless a supported TorchRL approach demonstrably cannot satisfy the requirements.

Do not replace the learner implementation.

If the prototype encounters a significant TorchRL incompatibility, report the issue and stop that implementation rather than building a large workaround.

### Performance comparison

Compare:

- Current grouped collector: CPU inference, CUDA learner
- Central synchronous collector: CPU inference, CUDA learner
- Central synchronous collector: CUDA inference, CUDA learner

Keep the same number of processes, environments, rollout frames, learner sample target, and optimizer workload.

Measure:

- Decisions/sec
- Inference calls
- Actual inference batch size
- Inference latency
- CPU/GPU transfer overhead
- Collection time
- Aggregation time
- Update time
- GPU utilization where measurable

Only retain centralized inference if it provides a meaningful improvement without compromising learning correctness.

---

# Phase 7 — Evaluate torch.compile and CUDA Graphs

**Priority: Optional, last**

Only investigate compilation if profiling identifies policy inference or optimization kernels as a significant bottleneck.

Do not compile the environment, simulation lifecycle, or dynamic reporting path.

Potential candidates:

- TorchRL actor forward pass
- Critic forward pass
- Repeated fixed-shape neural network computations

Investigate `torch.compile` first.

CUDA Graphs should be considered only where fixed-shape execution, memory ownership, and stochastic policy sampling can be handled safely.

Verify compatibility with:

- Grouped environment batch shapes
- PPO stochastic sampling
- MAPPO shared and independent actors
- LoRA adapters
- Actor weight synchronization
- CPU versus CUDA execution
- Checkpoint loading

Measure compilation/warmup cost separately from steady-state throughput.

Do not enable compilation by default.

If the measured benefit is negligible or the implementation introduces complexity, skip it.

---

# Phase 8 — Focused Correctness Validation

Reuse existing tests wherever possible.

Do not add a large new test suite.

Minimum required validation:

**PPO**

- Serial 1×1
- Grouped 2×2
- Grouped 2×4
- CPU collection + CUDA learner

**MAPPO**

- Serial 1×1
- Grouped 2×2
- Grouped 2×4
- Actual 2v2 hybrid PP/FTG scenario

Validate:

- Environment IDs and independent seeds
- Correct TensorDict shape contracts
- Correct physical decision accounting
- Correct active learner samples
- Episode resets and independent trajectories
- GAE and bootstrap boundaries
- Retired-agent and fixed-only continuation behavior
- PPO/MAPPO optimizer steps
- Actor weight synchronization
- No stale policy versions
- No missing race events
- Valid checkpoint save/load
- Clean shutdown

For MAPPO, preserve shared team rewards, centralized critic inputs, and per-agent activity masks.

Do not silently update stale test expectations. If an unrelated baseline test already fails, document it and leave it out of the optimization scope.

---

# Phase 9 — Prepare Final HPC Evaluation

Do not run the complete HPC benchmark matrix as part of this coding task.

Provide commands to execute it later.

The matrix should compare the existing grouped collector and optimized implementation under matched conditions.

### Initial PPO matrix

Use:

- PPO lap-completion pretrain scenario
- CPU collection
- CUDA learner
- Same map, physics, reward and observation configuration
- Fixed seed
- `n_steps=4096`
- `batch_size=512`
- `n_epochs=4`
- Evaluation disabled
- W&B disabled
- Detailed profiling disabled for throughput measurement

Candidate layouts:

| Workers | Envs/worker | Total envs | Frames/batch |
|---|---|---|---|
| 4 | 2 | 8 | 1024 |
| 8 | 1 | 8 | 1024 |
| 8 | 2 | 16 | 1024 |
| 8 | 4 | 32 | 1024 |
| 16 | 2 | 32 | 1024 |
| 8 | 2 | 16 | 512 |
| 8 | 2 | 16 | 2048 |
| 8 | 4 | 32 | 2048 |

Use a larger profiling budget than the existing 65,536-decision experiments.

Propose at least 262,144 decisions for the final shortlisted configurations, with repetitions to assess performance variability.

### MAPPO matrix

After PPO, prepare a smaller matrix using the actual:

`scenarios/mappo_2v2_completion_scratch.yaml`

Evaluate:

- 4×2
- 8×2
- 8×4

Tune MAPPO independently rather than assuming the fastest PPO configuration is also optimal for 2v2 racing.

Record joint environment decisions, active learner samples, and optimizer steps separately.

### Centralized inference

If implemented successfully, include matched centralized-inference comparisons with the best grouped collector.

Do not combine centralized inference, affinity changes, and compilation in the same initial comparison.

### Required final metrics

- Overall environment decisions/sec
- Steady-state environment decisions/sec
- Active learner samples/sec
- Physics steps/sec
- Collector startup time
- Collection wall time
- Aggregation wall time
- Learner update time
- Weight synchronization time
- Worker waiting and imbalance
- Actual optimizer steps
- Number of completed episodes
- Peak process memory, when available
- CUDA utilization/memory, when available
- Actual CPU affinity and process count

Distinguish overlapping timings from additive wall-clock stages.

Preserve existing `run_summary.json`, `update_metrics.csv`, and configuration snapshots.

Avoid unnecessary new output formats.

---

# Phase 10 — Completion Report

Provide a concise final response containing:

1. Files changed and reason for each change.
2. Before/after results for aggregation optimization.
3. Before/after results for reporting optimization.
4. Before/after results for tensor construction optimization.
5. Whether CPU affinity control was useful.
6. Whether centralized inference was implemented, rejected, or deferred, with supporting evidence.
7. Whether compilation was worth enabling.
8. PPO and MAPPO correctness validation results.
9. Known limitations or regressions.
10. Exact commands for the final HPC evaluation matrix.

Explicitly identify changes that improved performance versus changes retained only for investigation.

Remove unused experimental code that provides no demonstrated benefit, unless it is needed for a clearly isolated benchmark.

Do not modify README files or write additional documentation.

## Execution instructions

Work incrementally and avoid extensive agent-token usage.

**Implementation order:**

1. Establish narrow timing measurements and preserve the baseline.
2. Optimize aggregation.
3. Optimize reporting.
4. Optimize environment tensor construction.
5. Validate correctness and measure the combined implementation.
6. Investigate CPU affinity with minimal changes.
7. Prototype centralized inference only if justified.
8. Consider compilation only if profiling supports it.
9. Prepare final benchmark commands and report.

Do not begin a large centralized-inference rewrite while the aggregation and environment-construction optimizations remain unmeasured.

If a phase introduces a correctness regression or requires substantial architectural change, stop that phase, retain the last working implementation, and explain the trade-off.

**The final deliverable is a stable, optimized, reproducible TorchRL PPO/MAPPO training pipeline ready for HPC performance evaluation—not a new RL framework.**