# MARL racing

Reinforcement learning experiments for racing.

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
stages already separate simulation from MPC costs. A future fixed-action baseline
needs a small controller registered with `register_fixed_controller` (with `act`,
`reset`, and environment injection as needed), and a separate comparison scenario
selecting it for `car_2`/`car_3` with the existing wheel adapter. Production opponent
and learner configurations remain unchanged. Prediction, footprint, and traffic
collision costs within the compiled `_shoot()` kernel remain combined; timing
inside that kernel would require further instrumentation.
