#!/usr/bin/env bash
#SBATCH --job-name=marl-ray-tune
#SBATCH --partition=gpu
#SBATCH --account=q1wang
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=ray-tune-%j.log

# Submit from ~/marl_race:
#   sbatch scripts/hpc/ray_tune_2node.sbatch
#   sbatch --export=ALL,TRIAL_MODE=both scripts/hpc/ray_tune_2node.sbatch
# Reuse EXPERIMENT_ID with RESTORE=1 in a new allocation for recovery.
# Sweeps: SWEEP_SPEC=scripts/sweeps/ppo_lr_smoke.yaml (or --sweep-spec PATH).
# Optional: RAY_TUNE_{TOTAL_STEPS,CHECKPOINT_STEPS,TRIAL_TIMEOUT,WANDB_MODE},
# RAY_TUNE_WANDB_{PROJECT,ENTITY,GROUP}, RAY_TUNE_EVALUATION=1.
# ENABLE_DASHBOARD=1, HOLD_SECONDS=300 (up to 1800).
set -Eeuo pipefail
umask 077

REPO="$HOME/marl_race"
VENV="$HOME/envs/marl-race"
TRIAL_MODE="${TRIAL_MODE:-ppo}"
export RESTORE="${RESTORE:-0}"
export SWEEP_SPEC="${SWEEP_SPEC:-}"
# Read only launcher identity arguments; pass the original arguments to Python.
launcher_args=("$@")
for ((i=0; i<${#launcher_args[@]}; i++)); do
  case "${launcher_args[i]}" in
    --sweep-spec) SWEEP_SPEC="${launcher_args[i+1]:?Missing sweep path}"; i=$((i+1));;
    --sweep-spec=*) SWEEP_SPEC="${launcher_args[i]#*=}";;
    --experiment-id) EXPERIMENT_ID="${launcher_args[i+1]:?Missing experiment ID}"; i=$((i+1));;
    --experiment-id=*) EXPERIMENT_ID="${launcher_args[i]#*=}";;
  esac
done
ENABLE_DASHBOARD="${ENABLE_DASHBOARD:-0}"
HOLD_SECONDS="${HOLD_SECONDS:-0}"

[[ "$TRIAL_MODE" == ppo || "$TRIAL_MODE" == mappo || "$TRIAL_MODE" == both ]] || { echo "Invalid TRIAL_MODE" >&2; exit 2; }
[[ "$RESTORE" =~ ^[01]$ ]] || exit 2
[[ "$ENABLE_DASHBOARD" == 0 || "$ENABLE_DASHBOARD" == 1 ]] || exit 2
[[ "$HOLD_SECONDS" =~ ^[0-9]+$ ]] && (( HOLD_SECONDS <= 1800 )) || exit 2
[[ -d "$VENV" ]] || { echo "Missing environment $VENV" >&2; exit 2; }
cd "$REPO"
[[ "$(git branch --show-current)" == multiNode ]] || { echo "Expected multiNode branch" >&2; exit 2; }
[[ -f scripts/ray_tune_trials.py ]] || { echo "Missing scripts/ray_tune_trials.py" >&2; exit 2; }
# Do not train from changing source code; new, untracked scripts are okay.
git diff --quiet -- src configs scenarios run.py || { echo "Uncommitted training source changes" >&2; exit 2; }
git diff --cached --quiet -- src configs scenarios run.py || exit 2

CRUN="$(command -v crun || true)"
if [[ -z "$CRUN" ]]; then
  CRUN=/cm/shared/containers/python3/2025.1-py312/crun.python3
fi
[[ -x "$CRUN" ]] || { echo "Cannot execute $CRUN" >&2; exit 2; }
if [[ -z "${EXPERIMENT_ID:-}" && -n "$SWEEP_SPEC" ]]; then
  EXPERIMENT_ID="$("$CRUN" -p "$VENV" python3 -c \
    'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["experiment_id"])' "$SWEEP_SPEC")"
fi
export EXPERIMENT_ID="${EXPERIMENT_ID:-slurm_${SLURM_JOB_ID}}"
[[ "$EXPERIMENT_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2

export RAY_DISABLE_USAGE_STATS=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export RAY_TUNE_GIT_SHA="$(git rev-parse HEAD)"
export RAY_TUNE_JOB_ROOT="${RAY_TUNE_JOB_ROOT:-$REPO/outputs/ray_tune_${EXPERIMENT_ID}}"
mkdir -p "$RAY_TUNE_JOB_ROOT"
chmod 700 "$RAY_TUNE_JOB_ROOT"
# flock prevents two allocations using the same experiment concurrently.
exec 9>"$RAY_TUNE_JOB_ROOT/.allocation.lock"
flock -n 9 || { echo "Experiment already has an active allocation" >&2; exit 2; }
ATTEMPT_ROOT="$RAY_TUNE_JOB_ROOT/attempts/${SLURM_JOB_ID}"
mkdir -p "$ATTEMPT_ROOT"

# Ray 2.59 token authentication; all nodes see the shared home filesystem.
export RAY_AUTH_MODE=token
unset RAY_AUTH_TOKEN || true
export RAY_AUTH_TOKEN_PATH="$ATTEMPT_ROOT/.ray_auth_token"
[[ ! -e "$RAY_AUTH_TOKEN_PATH" ]] || { echo "Token file already exists" >&2; exit 2; }
trap 'rm -f -- "$RAY_AUTH_TOKEN_PATH"' EXIT
"$CRUN" -p "$VENV" python3 -c 'import secrets; print(secrets.token_urlsafe(48))' > "$RAY_AUTH_TOKEN_PATH"
chmod 600 "$RAY_AUTH_TOKEN_PATH"
[[ -s "$RAY_AUTH_TOKEN_PATH" ]] || exit 2

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#NODES[@]} -eq 2 && "${NODES[0]}" != "${NODES[1]}" ]] || exit 2
HEAD="${NODES[0]}"
WORKER="${NODES[1]}"
node_ip() { getent ahostsv4 "$1" | awk '$2=="STREAM" {print $1; exit}'; }
HEAD_IP="$(node_ip "$HEAD")"
WORKER_IP="$(node_ip "$WORKER")"
[[ -n "$HEAD_IP" && -n "$WORKER_IP" ]] || exit 2

# Identical port block on both nodes, unique to the Slurm job (modulo collisions possible).
BASE=$((15000 + (SLURM_JOB_ID % 350) * 120))
GCS_PORT=$BASE
CLIENT_PORT=$((BASE + 1))
DASHBOARD_PORT=$((BASE + 2))
NODE_PORT=$((BASE + 3))
OBJECT_PORT=$((BASE + 4))
METRICS_PORT=$((BASE + 5))
RUNTIME_PORT=$((BASE + 6))
AGENT_HTTP_PORT=$((BASE + 7))
AGENT_GRPC_PORT=$((BASE + 8))
MIN_WORKER_PORT=$((BASE + 20))
MAX_WORKER_PORT=$((BASE + 109))
export RAY_ADDRESS="$HEAD_IP:$GCS_PORT"
TEMP_DIR="/tmp/ray_${USER}_${SLURM_JOB_ID}"
DASHBOARD_FLAG=false
[[ "$ENABLE_DASHBOARD" == 1 ]] && DASHBOARD_FLAG=true

head_step=""
worker_step=""
driver_step=""
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  # Stop the driver first, then the cluster. Worker watchdogs terminate native
  # trainer process groups when Ray exits. Persistent experiment data stays put.
  [[ -z "$driver_step" ]] || kill "$driver_step" 2>/dev/null || true
  [[ -z "$driver_step" ]] || wait "$driver_step" 2>/dev/null || true
  [[ -z "$worker_step" ]] || kill "$worker_step" 2>/dev/null || true
  [[ -z "$head_step" ]] || kill "$head_step" 2>/dev/null || true
  [[ -z "$worker_step" ]] || wait "$worker_step" 2>/dev/null || true
  [[ -z "$head_step" ]] || wait "$head_step" 2>/dev/null || true
  rm -f -- "$RAY_AUTH_TOKEN_PATH"
  if (( rc != 0 )); then
    echo "ERROR: Ray Tune job failed (exit $rc)" >&2
    for f in "$ATTEMPT_ROOT/head.log" "$ATTEMPT_ROOT/worker.log" "$ATTEMPT_ROOT/driver.log"; do
      [[ -f "$f" ]] && { echo "=== $f ==="; tail -n 45 "$f"; } >&2 || true
    done
  fi
  echo "Ray Tune completed: exit=$rc; outputs=$RAY_TUNE_JOB_ROOT"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'Git SHA: %s\nHead: %s (%s)\nWorker: %s (%s)\nMode: %s\nOutputs: %s\n' \
  "$RAY_TUNE_GIT_SHA" "$HEAD" "$HEAD_IP" "$WORKER" "$WORKER_IP" "$TRIAL_MODE" "$RAY_TUNE_JOB_ROOT"
if [[ "$ENABLE_DASHBOARD" == 1 ]]; then
  echo "Dashboard on head: $HEAD at http://127.0.0.1:$DASHBOARD_PORT"
  echo "Token (do not share): $RAY_AUTH_TOKEN_PATH"
fi

srun_base=(srun --overlap --nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task=8 --gres=gpu:1)
run_on() { local node="$1"; shift; "${srun_base[@]}" --nodelist="$node" "$CRUN" -p "$VENV" "$@"; }
common_args=(
  --num-cpus=8 --num-gpus=1
  "--node-manager-port=$NODE_PORT"
  "--object-manager-port=$OBJECT_PORT"
  "--metrics-export-port=$METRICS_PORT"
  "--runtime-env-agent-port=$RUNTIME_PORT"
  "--dashboard-agent-listen-port=$AGENT_HTTP_PORT"
  "--dashboard-agent-grpc-port=$AGENT_GRPC_PORT"
  "--min-worker-port=$MIN_WORKER_PORT"
  "--max-worker-port=$MAX_WORKER_PORT"
  --disable-usage-stats --block
)

run_on "$HEAD" ray --version
"${srun_base[@]}" --nodelist="$HEAD" "$CRUN" -p "$VENV" \
  ray start --head --node-ip-address="$HEAD_IP" --port="$GCS_PORT" \
  --ray-client-server-port="$CLIENT_PORT" \
  --include-dashboard="$DASHBOARD_FLAG" --dashboard-host=127.0.0.1 \
  --dashboard-port="$DASHBOARD_PORT" --temp-dir="$TEMP_DIR" \
  "${common_args[@]}" > "$ATTEMPT_ROOT/head.log" 2>&1 &
head_step=$!

ready=0
for ((i=0; i<90; i++)); do
  if (echo >/dev/tcp/"$HEAD_IP"/"$GCS_PORT") >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$head_step" 2>/dev/null; then
    echo "Ray head step exited during startup" >&2
    exit 1
  fi
  sleep 2
done
(( ready == 1 )) || { echo "Timed out waiting for head GCS" >&2; exit 1; }

"${srun_base[@]}" --nodelist="$WORKER" "$CRUN" -p "$VENV" \
  ray start --address="$RAY_ADDRESS" --node-ip-address="$WORKER_IP" \
  --temp-dir="$TEMP_DIR" "${common_args[@]}" \
  > "$ATTEMPT_ROOT/worker.log" 2>&1 &
worker_step=$!

# Tune waits for both GPUs before launching trials. The driver is lightweight.
"${srun_base[@]}" --nodelist="$HEAD" "$CRUN" -p "$VENV" \
  python3 -u scripts/ray_tune_trials.py "$@" > "$ATTEMPT_ROOT/driver.log" 2>&1 &
driver_step=$!
echo "Driver log: $ATTEMPT_ROOT/driver.log"
wait "$driver_step"
driver_step=""

if (( HOLD_SECONDS > 0 )); then
  echo "Keeping authenticated Ray Dashboard available for ${HOLD_SECONDS}s"
  sleep "$HOLD_SECONDS"
fi
