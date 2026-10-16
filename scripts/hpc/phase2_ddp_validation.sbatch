#!/usr/bin/env bash
#SBATCH --job-name=marl-ddp-validate
#SBATCH --partition=reserved-q1wang
#SBATCH --qos=q1wang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --time=01:30:00
#SBATCH --output=phase2-ddp-%j.log

set -Eeuo pipefail
cd "$HOME/marl_race"

# Load the ODU Python 3.12 container (the login shell may be tcsh).
if command -v module >/dev/null 2>&1; then
  module load container_env/0.1 python3/2025.1-py312
fi
if ! command -v crun.python3 >/dev/null 2>&1; then
  echo "ODU Python container launcher not found; load container_env and python3/2025.1-py312 before sbatch" >&2
  exit 1
fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYGLET_HEADLESS=true
VENV="$HOME/envs/marl-race"
OUT="$PWD/outputs/phase2_validation_${SLURM_JOB_ID}"
mkdir -p "$OUT"

trap 'echo "FAILED near line $LINENO; see $OUT and phase2-ddp-${SLURM_JOB_ID}.log" >&2' ERR

echo "Git commit: $(git rev-parse --short HEAD)"
git status --short
echo "Results: $OUT"

run_ddp() {
  local name="$1"; shift
  echo "===== $name ====="
  srun --ntasks=1 --cpus-per-task=16 --gres=gpu:2 --cpu-bind=cores \
    crun.python3 -p "$VENV" python3 -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node=2 "$@" \
    2>&1 | tee "$OUT/${name}.log"
}

run_test() {
  local name="$1"; shift
  echo "===== $name ====="
  srun --ntasks=1 --cpus-per-task=16 --gres=gpu:2 --cpu-bind=cores \
    crun.python3 -p "$VENV" python3 -m pytest "$@" \
    2>&1 | tee "$OUT/${name}.log"
}

COMMON=(
  --no-wandb --no-render --torch-threads 1 --seed 42
  --total-steps 8192
  --set experiment.distributed=true
  --set experiment.parallel_envs=8
  --set experiment.envs_per_worker=2
  --set ++experiment.collection_backend=centralized
  --set algorithm.params.n_steps=2048
  --set algorithm.params.n_epochs=2
  --set algorithm.params.batch_size=512
  --set ++algorithm.params.frames_per_batch=1024
  --set evaluation.enabled=false
  --set checkpoint.every_updates=1
  --set checkpoint.every_steps=null
)

# Hardware communication and focused correctness checks.
run_ddp nccl scripts/check_distributed.py
run_test distributed_tests tests/test_distributed_training.py -q -ra
run_test cuda_placement tests/test_on_policy_pipeline.py tests/test_torchrl_mappo.py \
  -k 'cuda:0-cuda:1' -q -ra

# Rank-local collectors with synchronized learning and checkpointing.
run_ddp ppo run.py --scenario scenarios/ppo_lap_completion_pretrain.yaml \
  "${COMMON[@]}" --output-dir "$OUT/ppo"
test -s "$OUT/ppo/final.pt"

for sharing in shared independent; do
  run_ddp "mappo_${sharing}" run.py \
    --scenario scenarios/mappo_2v2_completion_scratch.yaml \
    "${COMMON[@]}" --set "network.actor.parameter_sharing=$sharing" \
    --output-dir "$OUT/mappo_${sharing}"
  test -s "$OUT/mappo_${sharing}/final.pt"
done

# PPO checkpoint initializes the frozen base policy for per-agent LoRA.
run_ddp mappo_lora run.py --scenario scenarios/mappo_2v2_completion_lora.yaml \
  "${COMMON[@]}" --pretrained-actor "$OUT/ppo/final.pt" \
  --output-dir "$OUT/mappo_lora"
test -s "$OUT/mappo_lora/final.pt"

# Recovery: same global horizon, with an intentional stop after 8192 decisions.
run_ddp resume_first run.py --scenario scenarios/ppo_lap_completion_pretrain.yaml \
  "${COMMON[@]}" --total-steps 16384 \
  --set ++experiment.stop_after_steps=8192 --output-dir "$OUT/resume"
test -s "$OUT/resume/latest.pt"

run_ddp resume_finish run.py --scenario scenarios/ppo_lap_completion_pretrain.yaml \
  "${COMMON[@]}" --total-steps 16384 --resume "$OUT/resume" \
  --output-dir "$OUT/resume"
test -s "$OUT/resume/final.pt"

# Verify final budgets and multiple synchronized updates, without a long benchmark.
crun.python3 -p "$VENV" python3 - "$OUT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for name in ("ppo", "mappo_shared", "mappo_independent", "mappo_lora", "resume"):
    summary = json.loads((root / name / "run_summary.json").read_text())
    expected = 16384 if name == "resume" else 8192
    assert summary["environment_steps"] >= expected, (name, summary["environment_steps"])
    assert summary["updates"] >= 2, (name, summary["updates"])
    print(f"PASS {name}: decisions={summary['environment_steps']} updates={summary['updates']}")
PY

echo "ALL PHASE 2 VALIDATION STAGES PASSED"
echo "Results: $OUT"
