#!/usr/bin/env bash
#SBATCH --job-name=ppo-lora-20k
#SBATCH --partition=reserved-q1wang
#SBATCH --account=q1wang
#SBATCH --qos=q1wang
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --output=ppo-lora-20k-%j.out
#SBATCH --error=ppo-lora-20k-%j.err
# One reserved node: 64 CPUs shared by four independent single-GPU tasks.
# Each task receives 16 CPUs and one GPU; the fifth family follows on task 0.
# mem=0 requests the node's configured memory.
set -Eeuo pipefail

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
RESUME=${RESUME:-0}
EXTRA=()
if [[ "$RESUME" == 1 ]]; then
  : "${RUN_ROOT:?Set RUN_ROOT to the original adapter experiment directory}"
  [[ -s "$RUN_ROOT/source.pt" ]] || { echo 'Missing frozen source snapshot' >&2; exit 3; }
  EXTRA=(--resume)
elif [[ "$RESUME" == 0 ]]; then
  RUN_ROOT=${RUN_ROOT:-"$REPO_DIR/outputs/ppo_lora_20k_${SLURM_JOB_ID}"}
  SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-"$REPO_DIR/pretrained_models/mlp_base_ppo_seed42_best.pt"}
  mkdir -p "$(dirname "$RUN_ROOT")"
  mkdir "$RUN_ROOT" # Refuse accidental reuse of an existing experiment.
  cp -- "$SOURCE_CHECKPOINT" "$RUN_ROOT/source.pt"
else
  echo 'RESUME must be 0 or 1' >&2; exit 2
fi
# Paths must be absolute and shared by all allocated compute nodes.
RUN_ROOT=$(cd "$RUN_ROOT" && pwd)
exec 9>"$RUN_ROOT/.allocation.lock"
flock -n 9 || { echo 'This experiment already has an active allocation' >&2; exit 3; }
printf 'Root=%s; tasks=%s; nodes=%s; CPUs/task=%s; one independent learner per GPU\n' \
  "$RUN_ROOT" "$SLURM_NTASKS" "$SLURM_JOB_NUM_NODES" "$SLURM_CPUS_PER_TASK"
printf 'Source=%s/source.pt; each family trains for 20000 episodes\n' "$RUN_ROOT"
# No preflight, pilot, or evaluation jobs. Normal checkpoint compatibility
# checks remain part of training initialization.
srun --ntasks="$SLURM_NTASKS" --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores --kill-on-bad-exit=0 \
  "$CRUN" -p "$ENV_PATH" python3 scripts/train_ppo_lora_families.py \
  --root "$RUN_ROOT" --seed 42 "${EXTRA[@]}"
