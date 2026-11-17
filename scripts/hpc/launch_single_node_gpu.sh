#!/usr/bin/env bash
# Shared runtime for single-node independent GPU jobs. Allocation comes from sbatch.
set -Eeuo pipefail
trap 'status=$?; if (( status != 0 )); then echo "Allocation ${SLURM_JOB_ID:-unknown} failed/interrupted (exit $status); recover with RESUME=1 RUN_ROOT=${RUN_ROOT:-unset}" >&2; fi' EXIT
trap 'echo "Slurm termination received; recover from latest.pt" >&2; exit 143' TERM
trap 'exit 130' INT

REPO_DIR=${REPO_DIR:-"$HOME/marl_race"}
ENV_PATH=${ENV_PATH:-"$HOME/envs/marl-race"}
cd "$REPO_DIR"
if command -v module >/dev/null 2>&1; then
  module load container_env/0.1 python3/2025.1-py312
fi
CRUN=${CRUN:-$(command -v crun.python3 || command -v crun || true)}
[[ -x "$CRUN" ]] || { echo 'Python container launcher unavailable' >&2; exit 2; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYGLET_HEADLESS=true PYTHONUNBUFFERED=1
export WEIGHT_SYNC_TIMEOUT=${WEIGHT_SYNC_TIMEOUT:-900}
[[ "${SLURM_JOB_NUM_NODES:-1}" == 1 ]] || { echo 'Requires a single node' >&2; exit 2; }
: "${SCENARIO:?Set SCENARIO to an experiment definition}"
RESUME=${RESUME:-0}
EXTRA=()
if [[ "$RESUME" == 1 ]]; then
  : "${RUN_ROOT:?Set RUN_ROOT to the original experiment directory}"
  [[ -d "$RUN_ROOT" ]] || { echo 'Missing recovery root' >&2; exit 3; }
  EXTRA+=(--resume)
elif [[ "$RESUME" == 0 ]]; then
  RUN_ROOT=${RUN_ROOT:-"$REPO_DIR/outputs/$(basename "$SCENARIO" .yaml)_${SLURM_JOB_ID}"}
  mkdir -p "$RUN_ROOT"
  if [[ -n "${SOURCE_CHECKPOINT:-}" ]]; then EXTRA+=(--source "$SOURCE_CHECKPOINT"); fi
  if [[ -n "${PROFILE:-}" ]]; then EXTRA+=(--profile "$PROFILE"); fi
  if [[ -n "${SEED:-}" ]]; then EXTRA+=(--seed "$SEED"); fi
  if [[ -n "${STEPS_PER_TASK:-}" ]]; then EXTRA+=(--steps-per-task "$STEPS_PER_TASK"); fi
  if [[ -n "${WORKERS:-}" ]]; then EXTRA+=(--workers "$WORKERS"); fi
  if [[ -n "${ENVS_PER_WORKER:-}" ]]; then EXTRA+=(--envs-per-worker "$ENVS_PER_WORKER"); fi
  if [[ -n "${RESERVED_CPUS:-}" ]]; then EXTRA+=(--reserved-cpus "$RESERVED_CPUS"); fi
else
  echo 'RESUME must be 0 or 1' >&2; exit 2
fi
RUN_ROOT=$(cd "$RUN_ROOT" && pwd)
exec 9>"$RUN_ROOT/.allocation.lock"
flock -n 9 || { echo 'Experiment already has an active allocation' >&2; exit 3; }
printf 'Root=%s; scenario=%s; tasks=%s; CPUs/task=%s\n' \
  "$RUN_ROOT" "$SCENARIO" "${SLURM_NTASKS:-1}" "${SLURM_CPUS_PER_TASK:-affinity}"
# srun inherits task, CPU and GPU requests from the allocation. Never remap CUDA
# devices: Slurm supplies task-local visibility, and the learner uses cuda:0.
srun --exact --gpu-bind=single:1 --cpu-bind=cores --kill-on-bad-exit=0 \
  "$CRUN" -p "$ENV_PATH" python3 scripts/launch_experiment.py \
  --scenario "$SCENARIO" --root "$RUN_ROOT" "${EXTRA[@]}" "$@"
