#!/usr/bin/env bash
# Invoke on Wahab in bash. Submits exactly one PPO job, with no dependent chain.
set -Eeuo pipefail
cd "${REPO_DIR:-$HOME/marl_race}"
export REPO_DIR="$PWD"
# Git is optional provenance, never a launch requirement.
SHA=$(git rev-parse HEAD 2>/dev/null) || SHA=unknown
git status --short 2>/dev/null || true
module load container_env/0.1 python3/2025.1-py312
CRUN=${CRUN:-$(command -v crun.python3 || command -v crun || true)}
[[ -x "$CRUN" ]] || { echo 'Missing Python container launcher' >&2; exit 2; }
export ENV_PATH="${ENV_PATH:-$HOME/envs/marl-race}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYGLET_HEADLESS=true
export RESUME=${RESUME:-0}
if [[ "$RESUME" == 1 ]]; then
  : "${RUN_ROOT:?Set RUN_ROOT to the original optimized experiment}"
  : "${STOP_AT:?Set the next absolute decision target, at most 120000000}"
  [[ -s "$RUN_ROOT/ppo/latest.pt" ]]
else
  RUN_ROOT=${RUN_ROOT:-"$PWD/outputs/ppo_opt_${SHA:0:7}_h2p5_s42_$(date -u +%Y%m%dT%H%M%SZ)_$$"}
  mkdir "$RUN_ROOT" # Atomic reservation; refuses an existing experiment.
fi
export RUN_ROOT
scontrol show partition reserved-q1wang | tee "$RUN_ROOT/partition.txt"
scontrol show node e7-w4125-01 | tee "$RUN_ROOT/node.txt"
squeue -w e7-w4125-01 -o '%.18i %.12u %.12T %.8C %.20b %.12m %.12L' | tee "$RUN_ROOT/existing_jobs.txt"
sacctmgr -nP show qos where name=q1wang format=Name,MaxWall,MaxTRES,MaxTRESPU,Flags | tee "$RUN_ROOT/qos.txt"
sacctmgr -nP show assoc where user="$USER" format=User,Account,Partition,QOS,MaxWall | tee "$RUN_ROOT/associations.txt"
# Use at most four days, reduced to reported partition/QoS walltime limits.
WALLTIME=$("$CRUN" -p "$ENV_PATH" python3 - "$RUN_ROOT" "${O7_WALLTIME:-4-00:00:00}" <<'PY'
import re, sys
from pathlib import Path
root = Path(sys.argv[1])
def seconds(value):
    if value in ('UNLIMITED', 'INFINITE', '', 'N/A'):
        return float('inf')
    days, _, clock = value.rpartition('-')
    parts = [int(p) for p in clock.split(':')]
    if len(parts) != 3:
        raise ValueError(f'Unrecognized Slurm walltime {value!r}')
    return int(days or 0)*86400 + parts[0]*3600 + parts[1]*60 + parts[2]
partition = re.search(r'\bMaxTime=(\S+)', (root/'partition.txt').read_text())
if partition is None:
    raise SystemExit('Cannot establish partition time limit')
qos = [line.split('|') for line in (root/'qos.txt').read_text().splitlines() if line.startswith('q1wang|')]
if len(qos) != 1:
    raise SystemExit('Cannot establish q1wang QoS limits')
limit = int(min(345600, seconds(sys.argv[2]), seconds(partition[1]), seconds(qos[0][1])))
if limit < 3600:
    raise SystemExit(f'Insufficient permitted walltime: {limit} seconds')
print(f'{limit//86400}-{limit%86400//3600:02}:{limit%3600//60:02}:{limit%60:02}')
PY
)
if [[ "$WALLTIME" == 4-00:00:00 ]]; then
  export STOP_AT=${STOP_AT:-120000000}
else
  # A single bounded segment; later recovery is explicit and retains the 120M LR horizon.
  export STOP_AT=${STOP_AT:-12288000}
  echo "Walltime limited to $WALLTIME; absolute stop=$STOP_AT, global budget=120000000"
fi
# Resolve and audit the exact launch command without starting training.
python_preflight="$RUN_ROOT/preflight.sh"
"$CRUN" -p "$ENV_PATH" python3 - "$python_preflight" <<'PY'
from pathlib import Path
import sys
script = Path('scripts/hpc/ppo_optimized_pretrain.sbatch').read_text()
start = script.index('CMD=(')
end = script.index("\nprintf 'Phase=")
prefix = '''set -Eeuo pipefail
SCENARIO=scenarios/ppo_lap_completion_pretrain.yaml
OUT="$RUN_ROOT/ppo"
SOURCE_SHA=$(git rev-parse HEAD 2>/dev/null) || SOURCE_SHA=unknown
TOTAL_STEPS=120000000
WORKERS=16
ENVS_PER_WORKER=2
RUN_ID="ppo_opt_${SOURCE_SHA:0:7}_h2p5_s42_$(basename "$RUN_ROOT")"
STOP_VALUE=null
(( STOP_AT == TOTAL_STEPS )) || STOP_VALUE="$STOP_AT"
EXTRA=()
[[ "$RESUME" == 0 ]] || EXTRA=(--resume "$OUT/latest.pt")
'''
Path(sys.argv[1]).write_text(prefix + script[start:end] + '''
"$CRUN" -p "$ENV_PATH" "${CMD[@]}" --resolve-config > "$RUN_ROOT/preflight_resolved.yaml"
"$CRUN" -p "$ENV_PATH" python3 scripts/check_ppo_optimized_pretrain.py "$RUN_ROOT/preflight_resolved.yaml" "${EXTRA[@]}"
''')
PY
export CRUN
bash "$python_preflight" | tee "$RUN_ROOT/preflight_contract_sha256.txt"
"$CRUN" -p "$ENV_PATH" python3 -c 'import torch, torchrl, wandb; assert torch.version.cuda is not None, "CPU-only PyTorch installation"; print("Python environment OK; CUDA runtime:", torch.version.cuda)'
LAUNCH=(sbatch --time="$WALLTIME" --chdir="$PWD" --export=ALL)
# Scheduler checks partition, account, QoS and resource constraints before submission.
"${LAUNCH[@]}" --test-only scripts/hpc/ppo_optimized_pretrain.sbatch 2>&1 | tee "$RUN_ROOT/slurm_test.txt"
JOB=$("${LAUNCH[@]}" --parsable scripts/hpc/ppo_optimized_pretrain.sbatch 2> "$RUN_ROOT/slurm_submission_warnings.txt")
printf '%s\n' "$JOB" | tee "$RUN_ROOT/job_id.txt"
cat "$RUN_ROOT/slurm_submission_warnings.txt"
printf 'SUBMITTED job=%s SHA=%s\nNode=e7-w4125-01 CPUs=64 GPU=1xL40S RAM=256G time=%s\nOutput=%s/ppo\nW&B group=ppo-optimized-%s-h2p5-s42\nLogs=%s/ppo-opt-%s.out and .err\n' \
  "$JOB" "$SHA" "$WALLTIME" "$RUN_ROOT" "${SHA:0:7}" "$PWD" "${JOB%%;*}"
echo 'Actual W&B ID is recorded in ppo/config_snapshot.json.'
