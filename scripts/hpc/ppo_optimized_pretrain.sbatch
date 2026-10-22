#!/usr/bin/env bash
#SBATCH --job-name=ppo-opt-h2p5
#SBATCH --partition=reserved-q1wang
#SBATCH --qos=q1wang
#SBATCH --nodelist=e7-w4125-01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=4-00:00:00
#SBATCH --output=ppo-opt-%j.out
#SBATCH --error=ppo-opt-%j.err
set -euo pipefail

# PPO only. Submit using submit_ppo_optimized_pretrain.sh from the repository root.
EXPECTED_SHA=b17b6721a95893a4eaad09807003584c9aae1d0e
PHASE=ppo
WORKERS=16
ENVS_PER_WORKER=2
SCENARIO=scenarios/ppo_lap_completion_pretrain.yaml
: "${RUN_ROOT:?Unique output root required}"
STOP_AT=${STOP_AT:-120000000}
RESUME=${RESUME:-0}
TOTAL_STEPS=120000000
(( STOP_AT > 0 && STOP_AT <= TOTAL_STEPS )) || { echo "Bad STOP_AT=$STOP_AT" >&2; exit 2; }
(( WORKERS > 0 && ENVS_PER_WORKER > 0 )) || exit 2

REPO_DIR=${REPO_DIR:-"$HOME/marl_race"}
ENV_PATH=${ENV_PATH:-"$HOME/envs/marl-race"}
cd "$REPO_DIR"
[[ "$(git branch --show-current)" == physics-correctness ]] || { echo "Expected checkout of physics-correctness branch" >&2; exit 2; }
[[ "$(git rev-parse HEAD)" == "$EXPECTED_SHA" ]] || { echo "Checkout changed: expected $EXPECTED_SHA" >&2; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo "Tracked files are modified; refusing mixed-code experiment" >&2; exit 2; }

if command -v module >/dev/null 2>&1; then
  module load container_env/0.1 python3/2025.1-py312
fi
CRUN=${CRUN:-$(command -v crun.python3 || command -v crun || true)}
[[ -x "$CRUN" ]] || { echo "Python container launcher unavailable" >&2; exit 2; }
export WEIGHT_SYNC_TIMEOUT=900
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYGLET_HEADLESS=true PYTHONUNBUFFERED=1
export F110_SUMMARY_EVERY=25

OUT="$RUN_ROOT/$PHASE"
RUN_ID="ppo_opt_${EXPECTED_SHA:0:7}_h2p5_s42_$(basename "$RUN_ROOT")"
mkdir -p "$RUN_ROOT"

# Recovery is explicit, exclusively from this 2.5 ms experiment.
EXTRA=()
if [[ "$RESUME" == 1 ]]; then
  [[ -s "$OUT/latest.pt" ]] || { echo "Missing latest.pt for explicit recovery" >&2; exit 3; }
  EXTRA=(--resume "$OUT/latest.pt")
elif [[ -e "$OUT" ]]; then
  echo "Output already exists; refusing overwrite: $OUT" >&2
  exit 3
fi

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
  --set environment.vehicle_params.max_integration_step=0.0025
  --set experiment.measure_stages=false
  --set experiment.measure_mpc=false
  --set experiment.measure_hybrid=false
  --set experiment.measure_services=true
  --set "++wandb.validation.git_sha=$EXPECTED_SHA"
  --set ++wandb.validation.integration_step_s=0.0025
  --set 'wandb.notes=Full 120M PPO pretraining with optimized physics and 2.5 ms integration.'
  --set "wandb.group=ppo-optimized-${EXPECTED_SHA:0:7}-h2p5-s42"
  --set "wandb.name=$RUN_ID"
  --wandb --no-render
  "${EXTRA[@]}")

printf 'Phase=%s target=%s/%s; workers=%s x %s; GPU=%s; cpus=%s\n' \
  "$PHASE" "$STOP_AT" "$TOTAL_STEPS" "$WORKERS" "$ENVS_PER_WORKER" \
  "${CUDA_VISIBLE_DEVICES:-unknown}" "${SLURM_CPUS_PER_TASK:-unknown}"
printf 'Repository=%s commit=%s\n' "$REPO_DIR" "$EXPECTED_SHA"
printf 'Output=%s\n' "$OUT"
printf 'Command:'; printf ' %q' "${CMD[@]}"; printf '\n'
[[ "${SLURM_CPUS_PER_TASK:-0}" == 64 ]]
[[ "$(scontrol show hostnames "$SLURM_JOB_NODELIST")" == e7-w4125-01 ]]
nvidia-smi -L
"$CRUN" -p "$ENV_PATH" "${CMD[@]}" --resolve-config > "$RUN_ROOT/resolved_job_${SLURM_JOB_ID}.yaml"
CONTRACT_SHA=$("$CRUN" -p "$ENV_PATH" python3 scripts/check_ppo_optimized_pretrain.py \
  "$RUN_ROOT/resolved_job_${SLURM_JOB_ID}.yaml" --gpu "${EXTRA[@]}")
[[ "$CONTRACT_SHA" =~ ^[0-9a-f]{64}$ ]]
CMD+=(--set "++wandb.validation.physics_contract_sha256=$CONTRACT_SHA")
"$CRUN" -p "$ENV_PATH" "${CMD[@]}" --resolve-config > "$RUN_ROOT/resolved_job_${SLURM_JOB_ID}.yaml"
sha256sum scripts/hpc/ppo_optimized_pretrain.sbatch scripts/check_ppo_optimized_pretrain.py > "$RUN_ROOT/launcher_${SLURM_JOB_ID}.sha256"
git status --short > "$RUN_ROOT/git_status_${SLURM_JOB_ID}.txt"
printf '%q ' "${CMD[@]}" > "$RUN_ROOT/command_${SLURM_JOB_ID}.txt"
printf '\n' >> "$RUN_ROOT/command_${SLURM_JOB_ID}.txt"
mkdir -p "$OUT"

# A failure propagates to SLURM; downstream afterok jobs do not run.
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --cpu-bind=cores \
  "$CRUN" -p "$ENV_PATH" "${CMD[@]}" 2>&1 | tee -a "$OUT/train.log"

[[ -s "$OUT/latest.pt" ]] || { echo "Missing recovery checkpoint after successful execution" >&2; exit 4; }
"$CRUN" -p "$ENV_PATH" python3 - "$OUT/run_summary.json" "$STOP_AT" <<'CHECK'
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
