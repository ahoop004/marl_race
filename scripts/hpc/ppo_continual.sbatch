#!/usr/bin/env bash
#SBATCH --job-name=ppo-continual
#SBATCH --partition=reserved-q1wang
#SBATCH --account=q1wang
#SBATCH --qos=q1wang
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=4-00:00:00
#SBATCH --output=ppo-continual-%j.out
#SBATCH --error=ppo-continual-%j.err
set -Eeuo pipefail
trap 'status=$?; if (( status != 0 )); then echo "Continual run interrupted; recover with RESUME=1 RUN_ROOT=${RUN_ROOT:-unset}" >&2; fi' EXIT
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
export WEIGHT_SYNC_TIMEOUT=900
PROFILE=${PROFILE:-pilot}
RESUME=${RESUME:-0}
EXTRA=()
if [[ "$RESUME" == 1 ]]; then
  : "${RUN_ROOT:?Set RUN_ROOT to the original continual experiment directory}"
elif [[ "$RESUME" == 0 ]]; then
  RUN_ROOT=${RUN_ROOT:-"$REPO_DIR/outputs/ppo_continual_${PROFILE}_${SLURM_JOB_ID}"}
  SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-"$REPO_DIR/pretrained_models/mlp_base_ppo_seed42_best.pt"}
  mkdir -p "$(dirname "$RUN_ROOT")"
  mkdir "$RUN_ROOT"
  cp -- "$SOURCE_CHECKPOINT" "$RUN_ROOT/.source.pt.tmp"
  chmod a-w "$RUN_ROOT/.source.pt.tmp"
  mv -- "$RUN_ROOT/.source.pt.tmp" "$RUN_ROOT/source.pt"
else
  echo 'RESUME must be 0 or 1' >&2; exit 2
fi
RUN_ROOT=$(cd "$RUN_ROOT" && pwd)
[[ -s "$RUN_ROOT/source.pt" ]] || { echo 'Missing frozen source snapshot' >&2; exit 3; }
exec 9>"$RUN_ROOT/.allocation.lock"
flock -n 9 || { echo 'Experiment already has an active allocation' >&2; exit 3; }
[[ "$SLURM_NTASKS" == 2 && "$SLURM_JOB_NUM_NODES" == 1 ]] || {
  echo 'Requires one node and two independent GPU tasks' >&2; exit 2;
}
if [[ "$RESUME" == 1 ]]; then EXTRA+=(--resume); fi
if [[ -n "${STEPS_PER_TASK:-}" ]]; then EXTRA+=(--steps-per-task "$STEPS_PER_TASK"); fi
printf 'Root=%s; profile=%s; one GPU per condition; CPUs/task=%s\n' "$RUN_ROOT" "$PROFILE" "$SLURM_CPUS_PER_TASK"
srun --nodes=1 --ntasks=2 --ntasks-per-node=2 --exact --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores --kill-on-bad-exit=0 \
  "$CRUN" -p "$ENV_PATH" python3 scripts/train_ppo_continual.py \
  --root "$RUN_ROOT" --profile "$PROFILE" --seed "${SEED:-42}" \
  --workers "${WORKERS:-8}" --envs-per-worker "${ENVS_PER_WORKER:-4}" "${EXTRA[@]}"
