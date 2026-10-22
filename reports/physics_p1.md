# P1 serial physics profiling baseline

Measured locally on 2026-10-21. The audited source is `40426ddcee4f5e9ba7699aec0743e4225212ed1b`; the working tree was clean before instrumentation. All work remains on `physics-correctness`, as requested. Profiling changes are uncommitted. No branch, commit, push, merge, history rewrite, optimization, training run, or HPC job was performed.

The exact measured instrumentation is saved in `outputs/physics_p1/corrected/source_snapshot/`, with a tracked-file patch and SHA-256 hashes in `results.json`. This distinguishes the audited commit from the uncommitted code used for measurement. The comparison archives contain the exact audited commit and original multiNode revision `7d65c76fadff9081ce59c4a42c7400b892adc92e`; they were extracted outside the checkout with `git archive`, without changing branches. Configs/scenarios are unchanged between those revisions.

## Implementation

| File | Change |
| --- | --- |
| `src/physics/dynamic_models.py` | Optional MF6.1, RHS/load-solve, solver-iteration, actuator-pair, RK4 substep, and dense-trace counters. |
| `src/physics/vehicle.py` | Pass the optional buffer through integration, contact actuator sampling, reset validation, and diagnostics. |
| `src/physics/collision_models.py` | Optional swept-query, AABB rejection, SAT/exterior separation, refinement-node/bisection, occupancy-cell/candidate counters. |
| `src/physics/simulaton.py` | Optional map-transform, occupancy-sweep, pair-sweep, resolution/rescan regions; event-search, repeated-pair and stop counts. |
| `src/physics/profiling.py` | Opt-in instance wrappers using the existing collection profiler callback protocol and exclusive timing accounting; restores methods/attributes on exit and errors. |
| `scripts/benchmark_physics.py` | Deterministic B1–B6 runner, paired profiling modes, isolated source-root comparison, exact parity evidence, resources, provenance, standalone kernels. |
| `tests/test_physics_profiling.py` | Six rollout parity tests plus exclusive timing/accounting and restoration checks. |
| `reports/physics_p1.md` | This measured baseline report. |

Existing `scripts/benchmark_collection.py` targets trainer/HPC experiments and was inspected but not invoked. The serial runner uses the same configuration/task factories without constructing a learner. `RaceEnv.py`, tire equations, reward logic, observations, and training algorithms are unchanged. Environment, LiDAR and task timing use temporary instance wrappers; no global patches or additional dependencies are needed.

Counters use one 19-element int64 buffer (152 bytes) per profiling session, with `None` as the disabled default. There are no clock reads in JIT loops. Load solves/MF calls include validation during measured contact stops, while episode resets are outside the measured session. Actuator samples count pairs, not individual actuator coordinates. Contact resolutions count operations, including repeated contact across steps, rather than unique crash events. Python broad-phase/event bookkeeping is timed in `physics_collisions`; JIT broad/SAT/refinement detail uses counters and separate kernel measurements.

## Workloads and method

| ID | Cars | Workload | Steps per repeat |
| --- | --- | --- | --- |
| B1 | 1 | Collision-free, direct physical actions | 40 |
| B2 | 2 | Collision-free, parallel separated lanes | 40 |
| B3 | 4 | Collision-free, four separated lanes | 40 |
| B4 | 2 | Head-on controlled contact | 12 |
| B5 | 4 | Two contact groups at different times, repeated pair checks | 12 |
| B6 | 4 | Production MAPPO 2v2 task: two policy action streams and two fixed hybrid PP/FTG controllers, full observations/rewards/snapshots | 40 |

All cases use `combined_slip_st` v2, the existing synthetic MF6.1 coefficients, nominal shared mu=1.0489, dt=0.05 s and maximum integration substep 0.001 s (50 substeps per moving vehicle step). Sensors retain 108 beams, FOV 4.7 rad, 2000 angular bins, epsilon 0.0001 m, 30 m physical maximum, and default 0.01 m range noise. The policy range normalization remains 10 m. Collision tolerances and responses retain their defaults.

B1–B5 use a deterministic 40 m square map with a closed one-cell border and 0.05 m occupancy resolution, seed 42. Free cases start at (5, 5+2*i), speed 2 m/s, steering 0.04*sin(0.2*step), and held wheel reference 40 rad/s. B4 starts at x=5 and 6.4, y=5, opposite headings, both at 2 m/s. B5 starts at x=[5,6.1,7.5,8.3], y=5, headings [0,0,0,pi], speeds [4,2,0,2] m/s. Contact cases use zero steering. Full initial states, configs, sensor settings and action formulas are in each results file.

B6 uses `mappo_2v2_completion_scratch`, circle_map, seed 42, map episode index 2 and spawn episode index 3. It fixes nominal grip for repeatability and truncates at 40 steps. Each active learner supplies normalized [0.02*sin(0.2*step), 0.1]; production action conversion and fixed-controller logic run normally. This measures the task pipeline, without neural inference, tensor adapter collection, GAE or optimization. Decisions/s means individual policy actor decisions, two per joint step in this rollout.

Nine repeats per mode, alternating disabled/enabled order. Each source tree is measured sequentially. The complete workload warms both Numba specializations before timing. Reset, parity hashing, evidence copies, export, and compilation/cache loading are excluded from steady-state step times. Startup/warmup includes construction, imports as needed, first rollouts, and parity capture; it is not a cold-JIT compilation-only measurement. JIT caches may already exist. Total timed calls include deterministic action construction, reported as unattributed overhead outside wrapped simulator/task regions.

Host: AMD Ryzen 5 5600G with Radeon Graphics; 12 allowed logical CPUs, inherited affinity. Python 3.12.3, NumPy 2.5.3, Numba 0.68.0, SciPy 1.18.1. These are short local serial samples; IQRs describe the nine repetitions, without claims of statistical significance.

## Steady-state throughput and overhead

Wall times and rates are medians of disabled runs. B1–B5 steps are synchronous Simulator steps; B6 steps are RaceEnv/task steps. One joint step advances all cars. Physics steps/s equals joint steps/s here because action_repeat=1. Per-vehicle costs divide joint wall time by car count. Profiling overhead is the ratio of median enabled/disabled elapsed time, minus one.

| Case | Wall/repeat (ms) | Joint steps/s (IQR) | ms/joint step | ms/vehicle step | Profiling overhead |
| --- | --- | --- | --- | --- | --- |
| B1 | 33.00 | 1212.1 (1194.5–1220.3) | 0.825 | 0.825 | +5.5% |
| B2 | 66.16 | 604.6 (587.6–606.9) | 1.654 | 0.827 | +5.9% |
| B3 | 130.57 | 306.4 (298.6–309.1) | 3.264 | 0.816 | +6.0% |
| B4 | 29.72 | 403.8 (401.5–405.7) | 2.476 | 1.238 | +4.4% |
| B5 | 56.38 | 212.9 (210.6–214.1) | 4.698 | 1.175 | +6.1% |
| B6 | 344.38 | 116.2 (114.6–117.3) | 8.609 | 2.152 | +5.9% |

B6 completes 232.3 individual policy decisions/s. B3 costs 3.96 times B1 per joint step, with 0.99 times the per-vehicle cost. B6 costs 2.64 times the four-car physics-only case; its fixed controllers, different motion and task work are included in that difference.

## Exclusive stage attribution

Percentages are medians of per-run exclusive wall shares in enabled runs, grouped without overlaps. Inclusive envelopes are saved separately and never summed. Timer/callback overhead is present in these measurements; use disabled runs for baseline throughput. Rounded medians need not sum to exactly 100%.

| Component | B1 % | B2 % | B3 % | B4 % | B5 % | B6 % | B6 ms/step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RK4 integration | 68.5 | 67.9 | 68.5 | 67.9 | 63.3 | 32.3 | 2.939 |
| LiDAR (rays, preparation, opponents, rescan overhead) | 11.8 | 15.0 | 16.0 | 13.3 | 18.0 | 5.8 | 0.528 |
| Collision queries and resolution | 8.7 | 7.3 | 6.7 | 11.9 | 12.6 | 3.1 | 0.283 |
| Fixed controllers | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 28.6 | 2.611 |
| Finish intersections | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 7.6 | 0.691 |
| Frenet, neighbors, track previews | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 4.1 | 0.376 |
| Observations, state copies, snapshots | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 7.5 | 0.691 |
| Reward composition | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.3 | 0.025 |
| Vehicle/simulator/environment/task bookkeeping | 9.0 | 8.9 | 8.1 | 6.2 | 5.7 | 10.1 | 0.917 |

`rk4_integration` includes tire evaluation, simultaneous load equilibrium, exact actuator sampling, dense coefficients, and integration validation. Those nested JIT costs are not individually timed in the rollout. `finish_intersections` includes plane/root candidate work and lap bookkeeping. The complete ungrouped exclusive seconds, percentages, inclusive envelopes and call counts are in the raw JSON.

## Operation counts per measured rollout

| Case | Vehicle substeps | MF6.1 calls | Load solves | Solver iterations | Iterations/solve | Actuator pairs | Trace segments |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | 2000 | 77592 | 8040 | 22716 | 2.83 | 6040 | 2000 |
| B2 | 4000 | 155184 | 16080 | 45432 | 2.83 | 12080 | 4000 |
| B3 | 8000 | 310368 | 32160 | 90864 | 2.83 | 24160 | 8000 |
| B4 | 1200 | 67344 | 4840 | 23992 | 4.96 | 3640 | 1200 |
| B5 | 2400 | 120012 | 9682 | 40642 | 4.20 | 7282 | 2400 |
| B6 | 8000 | 374124 | 32160 | 122742 | 3.82 | 24160 | 8000 |

MF6.1 counts include zero-load bracket endpoint calls. The solver iteration count excludes its two bracket evaluations; MF calls therefore equal 2*(iterations + 2*solves) for these nonzero-height configurations. A collision-free vehicle step performs 201 RHS/load solves, 151 actuator-pair samples, and 50 dense trace segments.

| Case | Swept queries / AABB rejects | Cells inspected / occupied candidates | SAT / exterior checks | Refinement nodes / splits | Pair / repeated queries | Resolution operations |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | 40 / 0 | 11682 / 0 | 0 / 80 | 40 / 0 | 0 / 0 | 0 |
| B2 | 120 / 40 | 23364 / 0 | 0 / 160 | 80 / 0 | 40 / 0 | 0 |
| B3 | 400 / 240 | 46728 / 0 | 0 / 320 | 160 / 0 | 240 / 0 | 0 |
| B4 | 36 / 3 | 6464 / 0 | 181 / 48 | 196 / 154 | 12 / 0 | 16 |
| B5 | 140 / 69 | 13200 / 0 | 385 / 96 | 410 / 312 | 92 / 20 | 34 |
| B6 | 866 / 510 | 32041 / 466 | 392 / 320 | 356 / 0 | 240 / 0 | 0 |

B5 exercises repeated event searches after an earlier contact group. B6 includes real occupied-cell candidates near track boundaries, while B1–B5 isolate pair contacts away from walls. Post-contact rescan call counts equal physical resolution operations in B4/B5; their inclusive durations are available under `post_contact_rescan` and their LiDAR work is attributed once to the nested sensor stages. No predictive stops occurred in these six baselines.

## Original revision and instrumentation-off comparison

| Case | Original steps/s | Corrected wall change vs original | Instrumentation-off wall change vs audited |
| --- | --- | --- | --- |
| B1 | 1432.0 | +18.1% | -0.4% |
| B2 | 704.8 | +16.6% | +3.1% |
| B3 | 350.0 | +14.2% | +1.3% |
| B4 | 1030.5 | +155.2% | +0.1% |
| B5 | 463.2 | +117.6% | +0.9% |
| B6 | 131.3 | +13.1% | -0.7% |

All six full output digests match the archived audited revision exactly. B1–B3 complete 40 collision-free steps with bit-identical eight-state trajectories and actuator values against the original simulator. The direct-action B4/B5 states also agree before the first contact step. Numerical trajectories are not compared after divergent contact handling.

In B4 both revisions first report collision at step index 4, but corrected contact is at 0.2049999714 s absolute simulated time: chassis x=[5.40999994,5.99000006], both stopped. The original endpoint is x=[5.5,5.9], overlapping, with both chassis speeds still 2 m/s; it has no contact timestamp API. B5 first corrected contact is at 0.1099999428 s, followed by the other pair at 0.2599999428 s. Endpoint flags in the original do not imply equivalent resolved states. These performance differences include the additional required contact work.

B6 has no collisions in either revision. Its fixed controllers consume corrected sensor/geometry observations, so its closed-loop action sequences can differ. The maximum position difference in this short run is about 0.0444 m; B6 is a system performance comparison, not an identical-input numerical integration comparison. Raw trajectories, per-step contact flags, participant indices, and timestamps (null when absent) are retained.

Small signed changes relative to the archived audited implementation are local timing variation plus disabled-hook overhead; these separate sequential runs cannot isolate a sub-percent code cost. They do not establish an optimization or regression. Enabled profiling overhead is reported separately above.

## Startup, CPU and memory

| Case | Startup/warmup s | Timed wall, all 18 repeats s | CPU % of one core | Lifetime peak RSS MiB |
| --- | --- | --- | --- | --- |
| B1 | 0.617 | 0.611 | 93.2 | 194.4 |
| B2 | 0.221 | 1.235 | 93.3 | 276.7 |
| B3 | 0.311 | 2.443 | 92.9 | 286.8 |
| B4 | 0.112 | 0.549 | 92.8 | 292.1 |
| B5 | 0.159 | 1.057 | 92.9 | 292.6 |
| B6 | 1.678 | 6.395 | 93.8 | 437.6 |

Corrected suite wall time was 20.99 s including construction, warmup, capture, resets and kernel microbenchmarks. CPU is process CPU time divided by timed wall time. RSS is the process lifetime high-water mark including imports, JIT caches/compiler allocations, and prior cases; it is not incremental per-environment memory and should not be divided by car count.

## Standalone kernel measurements

Seven repeats of 20,000 calls in compiled loops, after a separate warmup. Inputs vary among three representative moving states; these are synthetic kernel costs, not sampled production stage attribution. Trace bounds use 50 dense segments. RHS includes its nested tire/load work. Do not add these rows together or multiply them into rollout percentages: zero-load calls, solver inputs, Python dispatch and allocation patterns differ.

| Kernel | Median microseconds/call |
| --- | --- |
| mf61_single_axle | 0.203 |
| rhs_with_load_solve | 4.819 |
| actuator_pair_sample | 0.044 |
| dense_trace_segment | 0.117 |
| swept_aabb | 0.597 |
| sat_separation | 0.687 |

## Validation and next profiling task

Focused correctness and instrumentation validation: **328 passed, 1 skipped**, with four existing upstream TorchRL deprecation warnings. The skip remains the legacy lateral-TTC projection case. New tests verify exact complete outputs across all six enabled/disabled rollouts, trajectory/actuator/scan/contact time parity, full task rewards and time-limit termination, counter totals, exclusive timing, disabled-clock behavior, and restoration after exceptions. Independently archived audited runs reproduce all six complete output digests. `git diff --check` passes.

| Rank | Measured component | B6 wall share |
| --- | --- | --- |
| 1 | RK4 integration | 32.3% |
| 2 | Fixed controllers | 28.6% |
| 3 | Finish intersections | 7.6% |
| 4 | Observations, state copies, snapshots | 7.5% |
| 5 | LiDAR (rays, preparation, opponents, rescan overhead) | 5.8% |
| 6 | Frenet, neighbors, track previews | 4.1% |
| 7 | Collision queries and resolution | 3.1% |
| 8 | Reward composition | 0.3% |

The next profiling task should remain serial: sample compiled RHS/load-equilibrium work to separate MF6.1 math, repeated solver work and allocations, then use existing hybrid-controller detail hooks and inspect finish-plane/root work on the same B6 trace. RK4 is the first physics target because it dominates isolated driving and is the largest full-task component. The current data does not support prioritizing pair broad-phase checks over integration. Collision-heavy B4/B5 and wall candidates in B6 should remain correctness/performance guard cases. No optimization or parallel profiling has started; baseline acceptance and an optimization branch decision remain for review.

## Reproduction and artifacts

From the repository root:

```bash
python3 scripts/benchmark_physics.py --output outputs/physics_p1/corrected --steps 40 --repeats 9 --kernels
```

For each baseline, extract `git archive <SHA>` into a fresh temporary directory, then run the same harness:

```bash
python3 scripts/benchmark_physics.py --source-root /absolute/archive/path \
  --source-revision <SHA> --output /absolute/results/path --no-profile --steps 40 --repeats 9
```

Run the revisions sequentially without other benchmarks or tests sharing the measurement window. Temporary archive paths are recorded in metadata; they can be recreated from Git. The generated arena PNG/YAML, all nine raw repeats, exact digest/trajectories/events, environment/config values, dependency versions, source hashes, and the measured source snapshot are in `outputs/physics_p1/{corrected,audited,original}/`. `outputs/physics_p1/summary.json` contains the tabulated medians and grouped stage shares. These output files are local, under the repository’s existing ignored `outputs/` directory.
