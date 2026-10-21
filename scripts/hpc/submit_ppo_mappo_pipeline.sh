#!/usr/bin/env bash
# Safe preview: bash scripts/hpc/submit_ppo_mappo_pipeline.sh --dry-run
# Submit training and independent evaluations: bash scripts/hpc/submit_ppo_mappo_pipeline.sh --submit
set -Eeuo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
PIPELINE_CRUN="${CRUN:-$(command -v crun || true)}"
[[ -n "$PIPELINE_CRUN" ]] || PIPELINE_CRUN=/cm/shared/containers/python3/2025.1-py312/crun.python3
PIPELINE_VENV="${VENV:-$HOME/envs/marl-race}"
[[ -x "$PIPELINE_CRUN" && -d "$PIPELINE_VENV" ]] || { echo "Missing Wahab container/project environment" >&2; exit 2; }
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYGLET_HEADLESS=true
export RAY_TUNE_REPO="$REPO" VENV="$PIPELINE_VENV" CRUN="$PIPELINE_CRUN"
exec "$PIPELINE_CRUN" -p "$PIPELINE_VENV" python3 scripts/ppo_mappo_pipeline.py "$@"
