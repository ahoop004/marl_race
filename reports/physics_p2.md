# P2: controlled serial physics and task attribution

Source: `433e990959aa98a1ba40499a908dcbe89255ca24` on **`physics-correctness`**, clean at the start of P2. Measured 2026-10-21. P2 adds optional instrumentation and local replay tools; no optimization, branch change, commit, push, or training run was performed. Exact modified source files are saved under `outputs/physics_p2/final/source_snapshot/`; the results record source SHA-256 hashes and the command used. P1 remains in `reports/physics_p1.md`.

**Throughput here means serial environment/task throughput. It excludes neural-network inference, TensorDict collection, GAE, learner optimization, and policy-input generation. It is not end-to-end MAPPO training throughput.**

## Controlled design and representative trajectories

Two four-car traces each run for 600 steps at dt=0.05 s: **30 simulated seconds per map**, compared with P1’s two-second B6. The nonlinear model, 0.001 s maximum substep, MF6.1 coefficients, nominal shared grip 1.0489, geometry, sensors, collision/finish rules and production reward/observation components are preserved. The source scenario is `mappo_2v2_completion_scratch` with a fixed 600-step time limit and deterministic explicit starting poses. All four cars remain active until the final time-limit step; neither trace collides or stops early.

Existing hybrid PP/FTG controllers generate the two learner action streams during recording; commands are encoded through the existing normalized steering/wheel-acceleration contract. The two production fixed opponents also use hybrid PP/FTG. This is a representative deterministic driving input generator, not a trained neural policy. Its cost is outside timed replay. Circle starts cars 9 m apart along the centerline; Budapest starts them with arc offsets [0,-3,-11,-16] m. Both start at 5% of track length before applying those offsets. Seed=42, map episode index=2, spawn episode index=3. Full poses and resolved configs are retained in results.

| Trace | Distance per car (m) | Accumulated heading change per car (rad) | Speed median / p95 (m/s) | Pair-distance range (m) |
| --- | --- | --- | --- | --- |
| circle_map | 227.4, 222.4, 216.9, 211.4 | 4.08, 4.02, 3.93, 3.84 | 8.00 / 8.00 | 8.98–41.94 |
| Budapest_map | 163.3, 149.8, 134.3, 130.3 | 8.27, 8.75, 9.39, 8.55 | 4.42 / 8.00 | 3.00–48.76 |

The recorder saves the actual float32 physical joint commands, learner normalized actions, eight-state trajectories, sampled RHS inputs, and complete task-output digests. The same commands are replayed through four increasingly complete layers:

| Layer | Work performed |
| --- | --- |
| Simulator | Original Simulator.step on the recorded physical joint command. |
| RaceEnv | Original RaceEnv.step with recorded physical actions, retaining Frenet/preview/finish/lifecycle/observation work. |
| RaceTask, recorded opponents | Original RaceTask.step, real learner action conversion, rewards, observations and snapshots; only the fixed adapters return recorded physical actions. |
| RaceTask, live opponents | Original RaceTask.step and original fixed hybrid controllers; learner normalized inputs are identical to recording. |

Every layer is checked at every step for exact physical commands and physics-output/state/contact parity. Both task layers also match the complete original TaskStep digest at every step, including observations, rewards and termination. Detailed profiling is separately parity-checked. Validation hashing/assertions run in separate passes; they are absent from timed throughput and detailed-attribution passes. Construction, reset, recording, input generation, Numba compilation/cache loading and solver tracing/microbenchmark warmup are outside steady-state timings. Three unprofiled repeats per layer rotate layer order to reduce drift; results include individual step durations. Source trees and measured layers run serially.

## Matched layer costs

| Trace | Simulator steps/s (ms/step) | RaceEnv steps/s (ms/step) | Task with recorded opponents steps/s (ms/step) | Task with live opponents steps/s (ms/step) |
| --- | --- | --- | --- | --- |
| circle_map | 298.2 (3.354) | 219.0 (4.566) | 187.0 (5.349) | 145.4 (6.877) |
| Budapest_map | 253.1 (3.951) | 191.4 (5.224) | 166.3 (6.013) | 129.5 (7.723) |

Rates and per-step costs are medians of the three unprofiled runs. One step advances all four vehicles; action_repeat=1, so task decisions and physics joint-step counts agree. Individual learner decisions/s is twice the live-task step rate. These rates exclude the artificial learner hybrid drivers used to create the trace.

| Trace | RaceEnv increment (ms/step) | Task increment, recorded opponents (ms/step) | Live fixed-controller increment (ms/step) |
| --- | --- | --- | --- |
| circle_map | 1.221 | 0.795 | 1.529 |
| Budapest_map | 1.278 | 0.775 | 1.723 |

Increments are medians of paired repeat differences on the same trace. The first adds environment bookkeeping/features, the second adds task conversion/reward/observation/snapshot work, and the last replaces replay adapters with live fixed planning. The last difference includes the small cost difference between the live adapter and the replay stub. No subtraction of P1 B3 and B6 is used, and no across-map subtraction is interpreted as causal overhead.

## MF6.1 and load-transfer solving

During recording, bounded optional buffers capture the exact six chassis states and two actuator values at the four RK4 stages of the first integration substep every 20 environment steps, for all four cars. This produces 480 RHS samples per map; 256 evenly spaced samples are used offline. The compiled authoritative RHS can separately record each actual load-equilibrium axle-force query into a cleared 63x8 buffer. Tracing reproduces RHS and diagnostic force arrays bit-for-bit. It is disabled during throughput/profiling runs. There are no clocks inside the production JIT loops.

Separate compiled batches replay (1) the full RHS/load solve, (2) its exact axle-force query sequence, (3) its exact front/rear MF6.1 calls including zero-load bracket endpoints, and (4) the replay/checksum loop floor. Physics parameters remain runtime inputs, avoiding constant-folded parameter benchmarks. Seven repeats of eight batches are reported after warmup. All sampled inputs, query counts and repeat durations are retained.

| Trace | RHS including solve (us/sample) | Exact axle-force sequence (us/sample) | Exact MF6.1 sequence (us/sample) | Replay floor (us/sample) | MF/RHS cost ratio |
| --- | --- | --- | --- | --- | --- |
| circle_map | 3.024 | 2.986 | 1.839 | 0.017 | 60.8% |
| Budapest_map | 3.841 | 3.769 | 2.327 | 0.019 | 60.6% |

These are inclusive matched-input kernel workloads, not additive production stage percentages. The axle-force sequence includes MF6.1, normal-load/residual calculations, and diagnostic-array creation; the RHS also includes the safeguarded iteration logic, chassis preparation/assembly and validation. Small RHS-minus-force differences cannot reliably isolate solver control-flow cost because replay/checksum work and generated code differ. MF6.1 is a substantial cost, but attributing all remaining time to the secant/bisection updates would be unsupported. No approximations, coefficient changes, solver shortcuts or allocation optimizations were made.

| Trace | Whole-rollout load solves / MF calls | Whole-rollout mean iterations/solve | Sample iteration min / median / p95 / max | Sample force queries / MF calls | Sample zero-load MF calls |
| --- | --- | --- | --- | --- | --- |
| circle_map | 482400 / 5353140 | 3.55 | 1 / 3 / 6 / 8 | 1410 / 2820 | 512 |
| Budapest_map | 482400 / 6477564 | 4.71 | 1 / 4 / 6 / 8 | 1705 / 3410 | 512 |

Whole-rollout counts cover every RK stage plus final validation. Sample iteration quantiles describe the sparse first-substep samples, not every RHS call or a proven global worst case. The complete mean/call totals prevent interpreting this sparse sample histogram as full-run coverage.

## Existing hybrid PP/FTG internal timing

P2 installs the existing `profile_hybrid` callback on each fixed opponent during detailed live-task profiling and restores it afterward. Controller equations and branch logic are unchanged. Internal sequential timings exclude adapter conversion/callback aggregation; they are nested inside `fixed_controller` and are never added again to full-pipeline percentages. Timed replay-adapter cases produce no hybrid records.

| Internal hybrid stage | circle_map % of hybrid time | Budapest_map % of hybrid time |
| --- | --- | --- |
| map_preparation | 0.4 | 0.4 |
| pure_pursuit | 14.8 | 13.2 |
| speed_planning | 19.0 | 16.5 |
| lidar_preprocessing | 5.6 | 5.2 |
| traffic_prediction | 17.9 | 22.4 |
| path_checks | 35.7 | 28.9 |
| other | 0.7 | 0.6 |
| smoothing | 3.3 | 3.0 |
| gap_selection | 2.7 | 9.8 |

| Trace | PP / FTG calls | Blocked / no-gap calls | Emergency brakes | Mode switches |
| --- | --- | --- | --- | --- |
| circle_map | 1170 / 30 | 5 / 0 | 0 | 10 |
| Budapest_map | 1094 / 106 | 20 / 0 | 0 | 34 |

Each map has 1,200 recorded fixed-controller calls. The workload dependence matters: tight positions/curves can change path checks, traffic prediction and FTG work even with identical controller code. The longer traces therefore refine the earlier 28.6% estimate rather than treating it as a universal constant.

## Finish-line processing: candidate traversal versus actual root solves

| Trace | RK trace segments | Level tests / hull rejections | Root solver calls | Root time (ms total) | Finish time (ms total) | Roots as % of finish / pipeline |
| --- | --- | --- | --- | --- | --- | --- |
| circle_map | 120000 | 240000 / 239984 | 16 | 1.400 | 472.81 | 0.296% / 0.0306% |
| Budapest_map | 120000 | 240000 / 240000 | 0 | 0.000 | 501.64 | 0.000% / 0.0000% |

Each of 600 steps visits four active paths with 50 dense substeps, testing both the finish plane and hysteresis level. The measured cost is overwhelmingly traversal/projection, Bernstein hull tests, and surrounding Python/lap bookkeeping. The data does **not** support repeated polynomial solving as the source of the earlier 7.6% finish-region cost. Root work is small because the existing hull rejection works. This identifies a future investigation target; no loops were skipped, consolidated or vectorized, and finish chronology remains unchanged.

## Observation and snapshot copies

| Trace | Recursive arrays / bytes copied | Containers / memo hits | Recursive detach time (ms/step) | Recursive detach % of live pipeline |
| --- | --- | --- | --- | --- |
| circle_map | 37800 / 2640000 | 73200 / 4200 | 0.527 | 6.9% |
| Budapest_map | 37800 / 2640000 | 73200 / 4200 | 0.532 | 6.3% |

The copy counters measure actual recursive `_detach` copies, preserving memo aliasing and read-only flags. They exclude explicit observation-array copies and composer/state-buffer copies, which remain in the other observation/snapshot timing group below. Recursive detach timing includes Python container traversal and dataclass reconstruction, so it is not a raw memory-bandwidth measurement. The byte count alone cannot explain or attribute the whole observation/snapshot region. The new detach boundary also captures step-fact and per-decision info copying that P1 charged to task bookkeeping, so summing the two P2 copy/observation groups is not a like-for-like comparison with P1’s 7.5% category. No copy removal, sharing, ownership or mutability changes were introduced.

## Full pipeline stage shares and longer-run variation

Exclusive stage groups are disjoint. Nested root solves are included once in finish processing; internal hybrid stages are shown separately above. Detailed hooks add overhead, so baseline rates come from unprofiled repeats rather than these passes.

| Component | circle_map % | Budapest_map % |
| --- | --- | --- |
| RK4 integration | 35.0 | 38.0 |
| Fixed controllers | 19.6 | 19.6 |
| Finish processing (including roots) | 10.3 | 9.9 |
| Recursive state detach | 6.9 | 6.3 |
| Other observation/snapshot/copy work | 6.2 | 5.7 |
| Frenet and track previews | 4.4 | 4.0 |
| LiDAR | 6.2 | 6.1 |
| Collision queries and resolution | 3.1 | 2.8 |
| Rewards | 0.3 | 0.3 |
| Other bookkeeping | 7.6 | 6.9 |

The following windows use unprofiled per-step durations and show sustained behavior beyond startup. Each row covers 200 steps (10 simulated seconds). Median costs are taken across the three live-task repeats for that window.

| Trace | Simulated seconds | Live task ms/step | Median speed (m/s) | p95 absolute yaw rate (rad/s) |
| --- | --- | --- | --- | --- |
| circle_map | 0–10 | 7.465 | 6.95 | 0.163 |
| circle_map | 10–20 | 6.552 | 8.00 | 0.144 |
| circle_map | 20–30 | 6.655 | 8.00 | 0.150 |
| Budapest_map | 0–10 | 7.891 | 3.91 | 0.621 |
| Budapest_map | 10–20 | 7.717 | 4.20 | 0.835 |
| Budapest_map | 20–30 | 7.533 | 5.59 | 1.207 |

Two thirty-second traces demonstrate sustained driving, cornering and varying vehicle separation; they do not establish coverage of all racing maps, close-contact traffic, or learned-policy behaviors. P1’s short synthetic scenarios remain separate reproducible guard workloads.

## Resources, overhead and collision guards

| Trace | Detail-pass overhead vs median live run | CPU % of one core | Process lifetime peak RSS MiB |
| --- | --- | --- | --- |
| circle_map | +11.0% | 99.9 | 339.0 |
| Budapest_map | +9.6% | 100.0 | 451.0 |

Detail overhead uses one clean profiled pass versus three unprofiled repeats and is a local diagnostic, not a precise paired confidence estimate. It includes kernel counters, finish counters, copy counters, exclusive wrappers and the existing hybrid timing hooks. RHS/force sampling is off in these passes. RSS includes imports, JIT compilation/cache allocations and earlier runs, not incremental per-car memory. CPU is process CPU divided by summed timed step walls, with small outer-loop bookkeeping included in process CPU. Warmup/capture timings are retained separately and should not be used as throughput results.

| Guard | Median disabled steps/s | Physical resolution operations | Repeated pair queries | Adaptive splits | Profiling parity / archived parity |
| --- | --- | --- | --- | --- | --- |
| B4 | 436.2 | 16 | 0 | 154 | True / True |
| B5 | 228.8 | 34 | 20 | 312 | True / True |

B4 is the unchanged head-on contact workload; B5 contains two chronologically distinct contact groups and rescans/repeated pair queries. The disabled/enabled complete digests match, including trajectories, actuator states, scans, flags, participant indices and contact timestamps. The complete digests also match an independent archive of the P1 source. No performance speedup or regression threshold is inferred from short guard timings alone; correctness events and counts are the primary guard evidence.

## Validation, changed files and recommendation

Focused checks: **347 passed, 1 skipped**, with four existing TorchRL deprecation warnings. New tests check exact matched layers on both maps, detailed-hook parity and restoration, hybrid stage accounting, real finish-before-contact and contact-before-finish outcomes, and copy-counter memo/read-only behavior. All 1,200 long task steps independently reproduce bit-identical states and complete TaskStep digests using an archive of source `433e990959aa98a1ba40499a908dcbe89255ca24`; both collision guard digests also match. The manifest is `archived_source_parity.json`. `git diff --check` passes.

| File | P2 change |
| --- | --- |
| `src/physics/dynamic_models.py` | Optional bounded RK-stage/force-query capture for offline exact-input replay; defaults disabled. |
| `src/physics/vehicle.py` | Validate/pass the optional RHS capture buffer. |
| `src/physics/profiling.py` | Reuse exclusive callback timing for RaceEnv-only replay, detailed finish/copy counters, and existing hybrid hooks. |
| `src/env/centerline_state.py` | Optional candidate/root counters and actual polyroots timing; unchanged finish equations/order. |
| `src/tasks/race_task.py` | Optional recursive copy counts and a detachable-copy timing boundary; unchanged copy ownership semantics. |
| `scripts/benchmark_physics_p2.py` | Long deterministic recording, exact layer replay, controlled costs, whole-run/window metrics and matched kernel batches. |
| `scripts/validate_physics_replay.py` | Independent replay against committed source archives. |
| `tests/test_physics_p2.py` | Focused P2 parity, root chronology and copy-counter tests. |
| `reports/physics_p2.md` | This measured report. |

Priority after review: continue to treat repeated MF6.1/axle-force evaluation as the leading physics target, with diagnostic-array/normal-load work distinguished from solver control flow. For hybrid planning, investigate the measured path-check and traffic-prediction costs before assuming FTG search dominates. For finish processing, focus on per-segment traversal and hull tests rather than polynomial roots. Keep recursive container/copy cost distinct from byte-copy bandwidth. Any later optimization should replay these same two traces and retain B4/B5 and finish/contact chronology tests. No optimization branch or implementation has been started.

## Reproduction and artifacts

```bash
python3 scripts/benchmark_physics_p2.py --output outputs/physics_p2/final --steps 600 --repeats 3
```

Extract `git archive 433e990959aa98a1ba40499a908dcbe89255ca24` into an isolated temporary directory, then:

```bash
python3 scripts/validate_physics_replay.py --source-root /absolute/archive/path \
  --source-revision 433e990959aa98a1ba40499a908dcbe89255ca24 \
  --results outputs/physics_p2/final
```

Host: AMD Ryzen 5 5600G with Radeon Graphics; inherited affinity over 12 logical CPUs; Python 3.12.3; versions {'numpy': '2.5.3', 'numba': '0.68.0', 'scipy': '1.18.1'}. The suite took 181.4 s including recording, warmup, validation and kernel microbenchmarks (archived validation is a separate subsequent command). All results are local under the existing ignored `outputs/` directory: `results.json`, `summary.json`, both compressed trace NPZ files, `archived_source_parity.json`, arena map files and `source_snapshot/`. The raw JSON retains ungrouped exclusive stages, inclusive envelopes, hybrid context/stage records, counters, every repeated/step timing, configs and source provenance.
