#!/usr/bin/env bash
#SBATCH --job-name=ppo-lora-20k
#SBATCH --partition=reserved-q1wang
#SBATCH --account=q1wang
#SBATCH --qos=q1wang
#SBATCH --nodes=1
#SBATCH --nodelist=e7-w4125-01
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=4-00:00:00
#SBATCH --output=ppo-lora-20k-%j.out
#SBATCH --error=ppo-lora-20k-%j.err
# One reserved node: 160 CPUs and five GPUs, one concurrent task per family.
# Slurm supplies each task's CPU affinity and visible GPU; no device IDs assumed.
# mem=0 requests the node's configured memory.
# Request four days without a shorter fallback; check acceptance with
# sbatch --test-only before submission (partition/QOS limits are site policy).
set -Eeuo pipefail
trap 'status=$?; if (( status != 0 )); then echo "LoRA allocation ${SLURM_JOB_ID:-unknown} failed/interrupted (exit $status); recover with RESUME=1 RUN_ROOT=${RUN_ROOT:-unset}" >&2; fi' EXIT
trap 'echo "Slurm termination received; recover from latest.pt; partial episodes are discarded" >&2; exit 143' TERM
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
export WEIGHT_SYNC_TIMEOUT=900
RESUME=${RESUME:-0}
EXTRA=()
if [[ "$RESUME" == 1 ]]; then
  : "${RUN_ROOT:?Set RUN_ROOT to the original adapter experiment directory}"
  [[ -s "$RUN_ROOT/source.pt" ]] || { echo 'Missing frozen source snapshot' >&2; exit 3; }
  EXTRA=(--resume)
elif [[ "$RESUME" == 0 ]]; then
  RUN_ROOT=${RUN_ROOT:-"$REPO_DIR/outputs/ppo_lora_20k_v2_${SLURM_JOB_ID}"}
  SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-"$REPO_DIR/pretrained_models/mlp_base_ppo_seed42_best.pt"}
  mkdir -p "$(dirname "$RUN_ROOT")"
  mkdir "$RUN_ROOT" # Refuse accidental reuse of an existing experiment.
  cp -- "$SOURCE_CHECKPOINT" "$RUN_ROOT/.source.pt.tmp"
  chmod a-w "$RUN_ROOT/.source.pt.tmp"
  mv -- "$RUN_ROOT/.source.pt.tmp" "$RUN_ROOT/source.pt"
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
[[ "$SLURM_NTASKS" == 5 && "$SLURM_JOB_NUM_NODES" == 1 && "$SLURM_CPUS_PER_TASK" == 32 ]] || {
  echo 'This launcher requires one node, five tasks and 32 CPUs/task' >&2; exit 2;
}
# No preflight, pilot, or evaluation jobs. Normal checkpoint compatibility
# checks remain part of training initialization.
srun --nodes=1 --ntasks=5 --ntasks-per-node=5 --exact --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  --gpus-per-task=1 --gpu-bind=single:1 --cpu-bind=cores --kill-on-bad-exit=0 \
  "$CRUN" -p "$ENV_PATH" python3 scripts/train_ppo_lora_families.py \
  --root "$RUN_ROOT" --seed 42 "${EXTRA[@]}"
echo "All five families completed: $RUN_ROOT"
