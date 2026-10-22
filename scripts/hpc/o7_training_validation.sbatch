#!/usr/bin/env bash
#SBATCH --job-name=o7-training-validation
#SBATCH --partition=reserved-q1wang
#SBATCH --qos=q1wang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=o7-training-%j.log

# Bounded adaptation of phase2_ddp_validation.sbatch and the serial profile scenarios.
# Run PPO then MAPPO, 8192 decisions each; no resume, transfer, sweep or full training.
set -Eeuo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
EXPECTED_SHA=b17b6721a95893a4eaad09807003584c9aae1d0e
[[ "$(git branch --show-current)" == physics-correctness ]]
[[ "$(git rev-parse HEAD)" == "$EXPECTED_SHA" ]]
# Untracked validation helpers are expected; tracked experiment files must be pristine.
git diff --quiet HEAD -- .
if command -v module >/dev/null 2>&1; then
  module load container_env/0.1 python3/2025.1-py312
fi
CRUN="${CRUN:-$(command -v crun.python3 || true)}"
[[ -n "$CRUN" ]] || CRUN=/cm/shared/containers/python3/2025.1-py312/crun.python3
[[ -x "$CRUN" ]]
VENV="${O7_VENV:-$HOME/envs/marl-race}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYGLET_HEADLESS=true PYTHONUNBUFFERED=1
export WEIGHT_SYNC_TIMEOUT=900 F110_SUMMARY_EVERY=25 WANDB_MODE=online
OUT="$PWD/outputs/o7_training_smoke_${SLURM_JOB_ID}"
mkdir "$OUT"
trap 'echo "FAILED near line $LINENO; results: $OUT" >&2' ERR
git status --short > "$OUT/git_status.txt"
git rev-parse HEAD > "$OUT/git_sha.txt"
sha256sum scripts/hpc/o7_training_validation.sbatch scripts/verify_o7_training.py > "$OUT/validation_scripts.sha256"
SCRIPT_SHA=$(sha256sum scripts/hpc/o7_training_validation.sbatch | cut -d ' ' -f 1)
COMMON=(
  --no-render --wandb --torch-threads 1 --seed 42 --total-steps 8192
  --set environment.vehicle_params.max_integration_step=0.0025
  --set 'maps.available=[circle_map,Budapest_map]' --set 'maps.train=[circle_map,Budapest_map]'
  --set experiment.parallel_envs=2 --set experiment.envs_per_worker=2
  --set ++experiment.collection_backend=centralized
  --set execution.params.device=cuda --set execution.params.collection_device=cpu
  --set checkpoint.every_updates=1 --set checkpoint.every_steps=null
  --set logging.physics_episodes=true
  --set ++wandb.entity=ahoop004-old-dominion-university
  --set ++wandb.mode=online --set ++wandb.group=o7-training-validation-b17b672
  --set '++wandb.notes=Bounded O7 smoke test with 8192 decisions at 2.5 ms using serial profile algorithm settings.'
  --set "++wandb.validation.git_sha=$EXPECTED_SHA"
  --set ++wandb.validation.integration_step_s=0.0025
  --set "++wandb.validation.script_sha256=$SCRIPT_SHA"
)
for algorithm in ppo mappo; do
  SCENARIO="scenarios/${algorithm}_serial_profile.yaml"
  RUN_ID="o7_${algorithm}_${SLURM_JOB_ID}_s42_h2p5"
  CMD=(python3 run.py --scenario "$SCENARIO" "${COMMON[@]}"
       --run-id "$RUN_ID" --output-dir "$OUT/$algorithm" --set "++wandb.name=$RUN_ID")
  "$CRUN" -p "$VENV" "${CMD[@]}" --resolve-config > "$OUT/${algorithm}_preflight.yaml"
  CONTRACT_SHA=$("$CRUN" -p "$VENV" python3 scripts/verify_o7_training.py prepare \
    --scenario "$SCENARIO" --resolved "$OUT/${algorithm}_preflight.yaml")
  [[ "$CONTRACT_SHA" =~ ^[0-9a-f]{64}$ ]]
  CMD+=(--set "++wandb.validation.physics_contract_sha256=$CONTRACT_SHA")
  "$CRUN" -p "$VENV" "${CMD[@]}" --resolve-config > "$OUT/${algorithm}_resolved.yaml"
  printf '%q ' "${CMD[@]}" > "$OUT/${algorithm}_command.txt"
  printf '\n' >> "$OUT/${algorithm}_command.txt"
  srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK}" --gres=gpu:1 --cpu-bind=cores \
    "$CRUN" -p "$VENV" "${CMD[@]}" 2>&1 | tee "$OUT/${algorithm}.log"
  "$CRUN" -p "$VENV" python3 scripts/verify_o7_training.py verify "$OUT/$algorithm" \
    > "$OUT/${algorithm}_verification.json"
done
# Verify actual online records separately after both training runs finish.
"$CRUN" -p "$VENV" python3 scripts/verify_o7_training.py verify "$OUT/ppo" "$OUT/mappo" --wandb-api \
  > "$OUT/wandb_verification.json"
echo "Bounded validation finished: $OUT"
