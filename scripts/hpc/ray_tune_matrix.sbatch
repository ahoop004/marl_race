#!/usr/bin/env bash
#SBATCH --job-name=marl-profile
#SBATCH --partition=gpu
#SBATCH --account=q1wang
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=ray-profile-%j.log
# Generalized public-partition launcher for the existing native Ray Tune adapter.
# Set node/GPU/CPU counts with sbatch CLI options, not edits to this file.
set -Eeuo pipefail
umask 077

REPO="$HOME/marl_race"
VENV="$HOME/envs/marl-race"
RESTORE="${RESTORE:-0}"
GPUS_PER_NODE="${PROFILE_GPUS_PER_NODE:-1}"
ENABLE_DASHBOARD="${ENABLE_DASHBOARD:-0}"
PROFILE_GPU_MONITOR="${PROFILE_GPU_MONITOR:-1}"
[[ "$RESTORE" =~ ^[01]$ && "$GPUS_PER_NODE" =~ ^[1-4]$ && "$ENABLE_DASHBOARD" =~ ^[01]$ && "$PROFILE_GPU_MONITOR" =~ ^[01]$ ]] || exit 2
[[ -d "$VENV" ]] || { echo "Missing project environment" >&2; exit 2; }
[[ -n "${SWEEP_SPEC:-}" && -n "${EXPERIMENT_ID:-}" ]] || { echo "SWEEP_SPEC and EXPERIMENT_ID required" >&2; exit 2; }
[[ "$EXPERIMENT_ID" =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
cd "$REPO"
[[ "$(git branch --show-current)" == multiNode ]] || { echo "Expected multiNode branch" >&2; exit 2; }
# Reuse the same source integrity contract as the current T5 launcher.
git diff --quiet -- HEAD -- src configs scenarios scripts run.py || { echo "Uncommitted source files" >&2; exit 2; }
[[ -z "$(git ls-files --others --exclude-standard -- src configs scenarios scripts run.py)" ]] || { echo "Untracked source files" >&2; exit 2; }
CRUN="$(command -v crun || true)"
[[ -n "$CRUN" ]] || CRUN=/cm/shared/containers/python3/2025.1-py312/crun.python3
[[ -x "$CRUN" ]] || { echo "Cannot execute $CRUN" >&2; exit 2; }

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NODE_COUNT="${#NODES[@]}"
CPUS_PER_NODE="${SLURM_CPUS_PER_TASK:?SBATCH --cpus-per-task required}"
[[ "$NODE_COUNT" -ge 1 && "$NODE_COUNT" -le 8 ]] || { echo "Expected 1..8 nodes" >&2; exit 2; }
[[ "$CPUS_PER_NODE" =~ ^[0-9]+$ && "$CPUS_PER_NODE" -ge 1 ]] || exit 2
export RAY_EXPECTED_NODES="$NODE_COUNT"
export RAY_EXPECTED_GPUS="$(( NODE_COUNT * GPUS_PER_NODE ))"
export RAY_DISABLE_USAGE_STATS=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export RAY_TUNE_GIT_SHA="$(git rev-parse HEAD)"
export RAY_TUNE_JOB_ROOT="${RAY_TUNE_JOB_ROOT:-$REPO/outputs/ray_tune_${EXPERIMENT_ID}}"
mkdir -p "$RAY_TUNE_JOB_ROOT"
chmod 700 "$RAY_TUNE_JOB_ROOT"
exec 9>"$RAY_TUNE_JOB_ROOT/.allocation.lock"
flock -n 9 || { echo "Another allocation is using this experiment" >&2; exit 2; }
ATTEMPT_ROOT="$RAY_TUNE_JOB_ROOT/attempts/${SLURM_JOB_ID}"
mkdir -p "$ATTEMPT_ROOT"
export RAY_AUTH_MODE=token
unset RAY_AUTH_TOKEN || true
export RAY_AUTH_TOKEN_PATH="$ATTEMPT_ROOT/.ray_auth_token"
[[ ! -e "$RAY_AUTH_TOKEN_PATH" ]] || exit 2
trap 'rm -f -- "$RAY_AUTH_TOKEN_PATH"' EXIT
"$CRUN" -p "$VENV" python3 -c 'import secrets; print(secrets.token_urlsafe(48))' > "$RAY_AUTH_TOKEN_PATH"
chmod 600 "$RAY_AUTH_TOKEN_PATH"
[[ -s "$RAY_AUTH_TOKEN_PATH" ]] || exit 2

node_ip() { getent ahostsv4 "$1" | awk '$2=="STREAM" {print $1;exit}'; }
HEAD="${NODES[0]}"
HEAD_IP="$(node_ip "$HEAD")"
[[ -n "$HEAD_IP" ]] || { echo "Cannot resolve Ray head" >&2; exit 1; }
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

# Every Ray step uses the *same per-node* Slurm CPU/GPU allocation.
srun_base=(srun --overlap --nodes=1 --ntasks=1 --ntasks-per-node=1
           --cpus-per-task="$CPUS_PER_NODE" --gres="gpu:${GPUS_PER_NODE}")
common_args=(--num-cpus="$CPUS_PER_NODE" --num-gpus="$GPUS_PER_NODE"
  "--node-manager-port=$NODE_PORT" "--object-manager-port=$OBJECT_PORT"
  "--metrics-export-port=$METRICS_PORT" "--runtime-env-agent-port=$RUNTIME_PORT"
  "--dashboard-agent-listen-port=$AGENT_HTTP_PORT" "--dashboard-agent-grpc-port=$AGENT_GRPC_PORT"
  "--min-worker-port=$MIN_WORKER_PORT" "--max-worker-port=$MAX_WORKER_PORT"
  --disable-usage-stats --block)

steps=()
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  # Driver is last in steps; stop it before shutting down Ray.
  for ((i=${#steps[@]}-1; i>=0; i--)); do
    kill "${steps[i]}" 2>/dev/null || true
  done
  for p in "${steps[@]}"; do wait "$p" 2>/dev/null || true; done
  rm -f -- "$RAY_AUTH_TOKEN_PATH"
  if (( rc != 0 )); then
    echo "ERROR: profiling allocation failed, exit $rc" >&2
    for f in "$ATTEMPT_ROOT"/*.log; do [[ -f "$f" ]] && tail -n 25 "$f" >&2; done
  fi
  echo "Ray matrix allocation exited=$rc, output=$RAY_TUNE_JOB_ROOT"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

echo "Git=$RAY_TUNE_GIT_SHA job=$SLURM_JOB_ID nodes=$NODE_COUNT GPUs=$RAY_EXPECTED_GPUS CPUs/node=$CPUS_PER_NODE"
echo "Experiment=$EXPERIMENT_ID sweep=$SWEEP_SPEC outputs=$RAY_TUNE_JOB_ROOT"
"${srun_base[@]}" --nodelist="$HEAD" "$CRUN" -p "$VENV" ray --version
"${srun_base[@]}" --nodelist="$HEAD" "$CRUN" -p "$VENV" ray start \
  --head --node-ip-address="$HEAD_IP" --port="$GCS_PORT" \
  --ray-client-server-port="$CLIENT_PORT" --include-dashboard="$DASHBOARD_FLAG" \
  --dashboard-host=127.0.0.1 --dashboard-port="$DASHBOARD_PORT" \
  --temp-dir="$TEMP_DIR" "${common_args[@]}" > "$ATTEMPT_ROOT/head.log" 2>&1 &
steps+=("$!")
ready=0
for ((i=0; i<90; i++)); do
  if (echo >/dev/tcp/"$HEAD_IP"/"$GCS_PORT") >/dev/null 2>&1; then ready=1; break; fi
  kill -0 "${steps[0]}" 2>/dev/null || { echo "Ray head exited unexpectedly" >&2; exit 1; }
  sleep 2
done
(( ready == 1 )) || { echo "Ray head startup timeout" >&2; exit 1; }

for ((i=1; i<NODE_COUNT; i++)); do
  node="${NODES[i]}"
  ip="$(node_ip "$node")"
  [[ -n "$ip" ]] || exit 1
  "${srun_base[@]}" --nodelist="$node" "$CRUN" -p "$VENV" ray start \
    --address="$RAY_ADDRESS" --node-ip-address="$ip" --temp-dir="$TEMP_DIR" \
    "${common_args[@]}" > "$ATTEMPT_ROOT/worker_${i}.log" 2>&1 &
  steps+=("$!")
done

# Optional low-frequency GPU utilization samples; results live alongside Ray logs.
# Missing nvidia-smi is nonfatal; native perf/* counters still remain available.
if [[ "$PROFILE_GPU_MONITOR" == 1 ]]; then
  for ((i=0; i<NODE_COUNT; i++)); do
    "${srun_base[@]}" --nodelist="${NODES[i]}" "$CRUN" -p "$VENV" \
      nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used \
      --format=csv --loop-ms=5000 > "$ATTEMPT_ROOT/gpu_${i}.csv" 2>&1 &
    steps+=("$!")
  done
fi

"${srun_base[@]}" --nodelist="$HEAD" "$CRUN" -p "$VENV" \
  python3 -u scripts/ray_tune_trials.py --sweep-spec "$SWEEP_SPEC" \
  --trial-timeout 0 --checkpoint-every-steps 65536 \
  > "$ATTEMPT_ROOT/driver.log" 2>&1 &
driver="$!"
steps+=("$driver")
echo "Driver log: $ATTEMPT_ROOT/driver.log"
wait "$driver"
