#!/usr/bin/env bash
#SBATCH --job-name=marl-training
#SBATCH --partition=reserved-q1wang
#SBATCH --qos=q1wang
#SBATCH --nodelist=e7-w4125-01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/ahoop004/marl-train-%j.out
#SBATCH --error=/home/ahoop004/marl-train-%j.err
set -euo pipefail

# Submitted by submit_marl_full_chain.sh. One single-GPU TorchRL learner per job.
: "${PHASE:?PHASE must be ppo, lora, or finetune}"
: "${STOP_AT:?Absolute training stop required}"
: "${RUN_ROOT:?Output root required}"
: "${EXPECTED_SHA:?Pinned Git revision required}"
: "${WORKERS:?Number of CPU process collectors required}"
: "${ENVS_PER_WORKER:?Environments per collector required}"

case "$PHASE" in
  ppo)      SCENARIO=scenarios/ppo_lap_completion_pretrain.yaml ;;
  lora)     SCENARIO=scenarios/mappo_2v2_completion_lora.yaml ;;
  finetune) SCENARIO=scenarios/mappo_2v2_completion_pretrained.yaml ;;
  *) echo "Unknown PHASE=$PHASE" >&2; exit 2 ;;
esac

TOTAL_STEPS=120000000
(( STOP_AT > 0 && STOP_AT <= TOTAL_STEPS )) || { echo "Bad STOP_AT=$STOP_AT" >&2; exit 2; }
(( WORKERS > 0 && ENVS_PER_WORKER > 0 )) || exit 2

REPO_DIR=${REPO_DIR:-"$HOME/marl_race"}
ENV_PATH=${ENV_PATH:-"$HOME/envs/marl-race"}
cd "$REPO_DIR"
[[ "$(git branch --show-current)" == multiGPU ]] || { echo "Expected checkout of multiGPU branch" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "$EXPECTED_SHA" ]] || { echo "Checkout changed: expected $EXPECTED_SHA" >&2; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo "Tracked files are modified; refusing mixed-code experiment" >&2; exit 2; }

module load python3/2025.1-py312
export WEIGHT_SYNC_TIMEOUT=900
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYGLET_HEADLESS=true PYTHONUNBUFFERED=1
export F110_SUMMARY_EVERY=25

OUT="$RUN_ROOT/$PHASE"
PPO_BEST="$RUN_ROOT/ppo/best.pt"
PPO_FINAL="$RUN_ROOT/ppo/final.pt"
RUN_ID="marl_full_${PHASE}_${EXPECTED_SHA:0:7}_s42"
mkdir -p "$RUN_ROOT"

# Never silently overwrite an existing run; resume when a recovery checkpoint exists.
EXTRA=()
if [[ -f "$OUT/latest.pt" ]]; then
  EXTRA=(--resume "$OUT/latest.pt")
elif [[ "$PHASE" != ppo ]]; then
  [[ -s "$PPO_BEST" && -s "$PPO_FINAL" ]] || {
    echo "PPO must finish its full budget before $PHASE: missing best.pt or final.pt" >&2; exit 3;
  }
  EXTRA=(--pretrained-actor "$PPO_BEST")
elif [[ -d "$OUT" && -n "$(ls -A "$OUT")" ]]; then
  echo "Nonempty PPO directory has no latest.pt; refusing to overwrite: $OUT" >&2
  exit 3
fi
if [[ "$PHASE" != ppo && ${#EXTRA[@]} -gt 0 && "${EXTRA[0]}" == "--pretrained-actor" && -d "$OUT" && -n "$(ls -A "$OUT")" ]]; then
  echo "Nonempty MAPPO directory has no latest.pt; refusing to overwrite: $OUT" >&2
  exit 3
fi
mkdir -p "$OUT"

# The stop is diagnostic/execution-only; the LR and training budget remain 120M.
if (( STOP_AT < TOTAL_STEPS )); then
  STOP_VALUE="$STOP_AT"
else
  STOP_VALUE=null
fi

CMD=(python3 run.py
  --scenario "$SCENARIO"
  --output-dir "$OUT"
  --run-id "$RUN_ID"
  --total-steps "$TOTAL_STEPS"
  --seed 42
  --torch-threads 1
  --set "experiment.parallel_envs=$WORKERS"
  --set "experiment.envs_per_worker=$ENVS_PER_WORKER"
  --set "++experiment.stop_after_steps=$STOP_VALUE"
  --set "execution.params.device=cuda"
  --set "execution.params.collection_device=cpu"
  --set "wandb.group=marl-full-ppo-mappo-${EXPECTED_SHA:0:7}"
  --set "wandb.name=$RUN_ID"
  --wandb --no-render
  "${EXTRA[@]}")

printf 'Phase=%s target=%s/%s; workers=%s x %s; GPU=%s; cpus=%s\n' \
  "$PHASE" "$STOP_AT" "$TOTAL_STEPS" "$WORKERS" "$ENVS_PER_WORKER" \
  "${CUDA_VISIBLE_DEVICES:-unknown}" "${SLURM_CPUS_PER_TASK:-unknown}"
printf 'Repository=%s commit=%s\n' "$REPO_DIR" "$EXPECTED_SHA"
printf 'Output=%s\n' "$OUT"
printf 'Command:'; printf ' %q' "${CMD[@]}"; printf '\n'
nvidia-smi -L

# A failure propagates to SLURM; downstream afterok jobs do not run.
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
  crun -p "$ENV_PATH" "${CMD[@]}" 2>&1 | tee -a "$OUT/train.log"

[[ -s "$OUT/latest.pt" ]] || { echo "Missing recovery checkpoint after successful execution" >&2; exit 4; }
python3 - "$OUT/run_summary.json" "$STOP_AT" <<'CHECK'
import json, sys
with open(sys.argv[1]) as f:
    summary = json.load(f)
actual, target = int(summary.get("environment_steps", 0)), int(sys.argv[2])
if actual < target:
    raise SystemExit(f"Training ended prematurely: {actual} < {target}")
print(f"Verified completed environment decisions: {actual} >= {target}")
CHECK
if (( STOP_AT == TOTAL_STEPS )); then
  [[ -s "$OUT/final.pt" ]] || { echo "Missing final.pt (full budget not reached)" >&2; exit 4; }
  [[ -s "$OUT/best.pt" ]] || { echo "Missing best.pt (check selection evaluation)" >&2; exit 4; }
fi
printf 'SUCCESS phase=%s stop=%s\n' "$PHASE" "$STOP_AT"
