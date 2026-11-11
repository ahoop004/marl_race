#!/usr/bin/env bash
#SBATCH --job-name=ppo-circle-20k
#SBATCH --partition=reserved-q1wang
#SBATCH --qos=q1wang
#SBATCH --nodelist=e7-w4125-01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=4-00:00:00
#SBATCH --output=ppo-circle-20k-%j.out
#SBATCH --error=ppo-circle-20k-%j.err
set -euo pipefail

REPO_DIR=${REPO_DIR:-"$HOME/marl_race"}
ENV_PATH=${ENV_PATH:-"$HOME/envs/marl-race"}
cd "$REPO_DIR"
if command -v module >/dev/null 2>&1; then
  module load container_env/0.1 python3/2025.1-py312
fi
CRUN=${CRUN:-$(command -v crun.python3 || command -v crun || true)}
[[ -x "$CRUN" ]] || { echo 'Python container launcher unavailable' >&2; exit 2; }
export WEIGHT_SYNC_TIMEOUT=900
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYGLET_HEADLESS=true PYTHONUNBUFFERED=1
OUT=${RUN_DIR:-"$REPO_DIR/pretrained_models/circle_ppo_seed42_20k"}
RESUME=${RESUME:-0}
EXTRA=()
if [[ "$RESUME" == 1 ]]; then
  [[ -s "$OUT/latest.pt" ]] || { echo "Missing recovery checkpoint: $OUT/latest.pt" >&2; exit 3; }
  EXTRA=(--resume "$OUT/latest.pt")
elif [[ "$RESUME" != 0 || -e "$OUT" ]]; then
  echo "Use RESUME=1 to recover an existing run, or choose a new RUN_DIR: $OUT" >&2
  exit 3
fi
CMD=(python3 run.py --scenario scenarios/ppo_circle_pretrain_20k.yaml
  --episodes 20000 --seed 42 --output-dir "$OUT" --no-render "${EXTRA[@]}")
# Resolve and validate without starting collection. Hyperparameters and collector
# layout come directly from the usual circle-track pretraining scenario.
mkdir -p "$OUT"
"$CRUN" -p "$ENV_PATH" "${CMD[@]}" --resolve-config > "$OUT/launch_resolved_${SLURM_JOB_ID}.yaml"
printf '%q ' "${CMD[@]}" > "$OUT/command_${SLURM_JOB_ID}.txt"
printf '\n' >> "$OUT/command_${SLURM_JOB_ID}.txt"
nvidia-smi -L
srun --ntasks=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-64}" --cpu-bind=cores \
  "$CRUN" -p "$ENV_PATH" "${CMD[@]}" 2>&1 | tee -a "$OUT/train.log"

# Publish the selected best checkpoint only after the episode budget completes.
"$CRUN" -p "$ENV_PATH" python3 - "$OUT" "$REPO_DIR/pretrained_models/mlp_base_ppo_seed42_best.pt" <<'PY'
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

run, target = map(Path, sys.argv[1:])
summary = json.loads((run / 'run_summary.json').read_text())
completed = int(summary['completed_episodes'])
if completed < 20000:
    raise SystemExit(f'Incomplete training: {completed}/20000 episodes; model not exported')
for name in ('best.pt', 'final.pt', 'latest.pt'):
    if not (run / name).is_file():
        raise SystemExit(f'Missing {name}; model not exported')
# Copy atomically; keep all original run checkpoints and logs.
fd, temporary = tempfile.mkstemp(prefix='.circle_ppo_export_', suffix='.pt', dir=target.parent)
os.close(fd)
try:
    shutil.copyfile(run / 'best.pt', temporary)
    os.replace(temporary, target)
finally:
    Path(temporary).unlink(missing_ok=True)
print(f'Completed {completed} episodes; exported selected model: {target}')
PY
