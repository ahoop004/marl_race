#!/usr/bin/env bash
#SBATCH --job-name=marl-ray-tune
#SBATCH --time=04:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=ray-tune-%j.log
# Submit with one task per node. Set CPU, GPU, and partition options with sbatch.
set -Eeuo pipefail
umask 077

REPO="${RAY_TUNE_REPO:-$PWD}"
RESTORE="${RESTORE:-0}"
ENABLE_DASHBOARD="${ENABLE_DASHBOARD:-0}"
PROFILE_GPU_MONITOR="${PROFILE_GPU_MONITOR:-1}"
HOLD_SECONDS="${HOLD_SECONDS:-0}"
[[ "$RESTORE" =~ ^[01]$ && "$ENABLE_DASHBOARD" =~ ^[01]$ && "$PROFILE_GPU_MONITOR" =~ ^[01]$ ]] || exit 2
[[ "$HOLD_SECONDS" =~ ^[0-9]+$ ]] && (( 10#$HOLD_SECONDS <= 1800 )) || exit 2
cd "$REPO"
REPO="$PWD"
[[ -f scripts/ray_tune_trials.py ]] || { echo "Run from the repository or set RAY_TUNE_REPO" >&2; exit 2; }
git diff --quiet -- HEAD -- src configs scenarios scripts run.py || { echo "Uncommitted source files" >&2; exit 2; }
[[ -z "$(git ls-files --others --exclude-standard -- src configs scenarios scripts run.py)" ]] || { echo "Untracked source files" >&2; exit 2; }

CRUN="${CRUN:-$(command -v crun.python3 || true)}"
wahab_crun=/cm/shared/containers/python3/2025.1-py312/crun.python3
if [[ -z "$CRUN" && -x "$wahab_crun" ]]; then CRUN="$wahab_crun"; fi
runtime=()
python_exec=python3
ray_exec=ray
if [[ -n "$CRUN" ]]; then
  VENV="${VENV:-$HOME/envs/marl-race}"
  [[ -x "$CRUN" && -d "$VENV" ]] || { echo "Cannot use $CRUN with environment $VENV" >&2; exit 2; }
  runtime=("$CRUN" -p "$VENV")
else
  VENV="${VENV:-${VIRTUAL_ENV:-}}"
  if [[ -z "$VENV" && -x "$REPO/.venv/bin/python" ]]; then VENV="$REPO/.venv"; fi
  if [[ -n "$VENV" ]]; then
    python_exec="$VENV/bin/python"
    ray_exec="$VENV/bin/ray"
  fi
fi

pipeline_action() {
  "${runtime[@]}" "$python_exec" scripts/ppo_mappo_pipeline.py --root "$PIPELINE_ROOT" \
    --stage "$PIPELINE_STAGE" --stage-action "$1"
}
pipeline_interrupted=0
pipeline_outcome() {
  local rc="$1"
  if [[ -n "${PIPELINE_ROOT:-}" && "$rc" != 0 ]]; then
    if (( pipeline_interrupted )); then
      pipeline_action interrupted || true
      if (( ${SLURM_RESTART_COUNT:-0} < PIPELINE_MAX_RESTARTS )); then
        echo "Requeuing incomplete $PIPELINE_STAGE stage, job=$SLURM_JOB_ID" >&2
        # Requeue while RUNNING: afterok retains the same unfinished parent ID.
        # An actual failed/timeout parent cannot revive an afterok dependency.
        scontrol requeue "$SLURM_JOB_ID" || { pipeline_action failed || true; return 1; }
      else
        echo "Pipeline retry limit reached" >&2
        pipeline_action failed || true
      fi
    else
      pipeline_action failed || true
    fi
  fi
  return "$rc"
}
if [[ -n "${PIPELINE_ROOT:-}" ]]; then
  pipeline_exports="$(pipeline_action environment)"
  eval "$pipeline_exports"
  if [[ "$PIPELINE_STAGE" == team || "$PIPELINE_STAGE" == solo ]]; then
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYGLET_HEADLESS=true
    exec 9>"$PIPELINE_ROOT/.${PIPELINE_STAGE}_allocation.lock"
    flock -n 9 || { echo "Evaluation allocation already active" >&2; exit 2; }
    evaluation_pid=""
    evaluation_cleanup() {
      local rc=$?
      trap - EXIT TERM INT USR1
      if [[ -n "$evaluation_pid" ]]; then
        kill -- "-$evaluation_pid" 2>/dev/null || true
        wait "$evaluation_pid" 2>/dev/null || true
      fi
      pipeline_outcome "$rc"
      exit "$?"
    }
    trap evaluation_cleanup EXIT
    trap 'pipeline_interrupted=1; exit 75' USR1
    trap 'exit 143' TERM
    trap 'exit 130' INT
    setsid "${runtime[@]}" "$python_exec" scripts/ppo_mappo_pipeline.py \
      --root "$PIPELINE_ROOT" --stage "$PIPELINE_STAGE" --stage-action evaluate &
    evaluation_pid="$!"
    wait "$evaluation_pid"
    exit 0
  fi
fi
launch_exports="$("${runtime[@]}" "$python_exec" scripts/ray_tune_trials.py --slurm-environment "$@")"
eval "$launch_exports"
export RAY_DISABLE_USAGE_STATS=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export SLURM_EXPORT_ENV=ALL
unset CUDA_VISIBLE_DEVICES SLURM_STEP_GPUS  # Slurm sets task-local GPU visibility.
[[ "${SLURM_HET_SIZE:-1}" == 1 ]] || { echo "Heterogeneous Slurm job groups are unsupported" >&2; exit 2; }
NODE_LIST="$(scontrol show hostnames "${SLURM_JOB_NODELIST:?Run with sbatch}")"
[[ -n "$NODE_LIST" ]] || { echo "No allocated Slurm nodes" >&2; exit 2; }
mapfile -t NODES <<< "$NODE_LIST"
NODE_COUNT="${#NODES[@]}"
REQUESTED_CPUS="${SLURM_CPUS_PER_TASK:-1}"
[[ "$REQUESTED_CPUS" =~ ^[1-9][0-9]*$ ]] || exit 2

mkdir -p "$RAY_TUNE_JOB_ROOT"
chmod 700 "$RAY_TUNE_JOB_ROOT"
exec 9>"$RAY_TUNE_JOB_ROOT/.allocation.lock"
flock -n 9 || { echo "Another allocation is using this experiment" >&2; exit 2; }
ATTEMPT_ROOT="$RAY_TUNE_JOB_ROOT/attempts/${SLURM_JOB_ID}"
if [[ -n "${PIPELINE_ROOT:-}" ]]; then ATTEMPT_ROOT="${ATTEMPT_ROOT}_r${SLURM_RESTART_COUNT:-0}"; fi
mkdir -p "$ATTEMPT_ROOT"
export RAY_AUTH_MODE=token
unset RAY_AUTH_TOKEN || true
export RAY_AUTH_TOKEN_PATH="$ATTEMPT_ROOT/.ray_auth_token"

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

# Steps inherit the job's requested GRES, including typed GPU requests.
srun_base=(srun --overlap --immediate=60 --nodes=1 --ntasks=1 --ntasks-per-node=1
           --cpus-per-task="$REQUESTED_CPUS" --cpu-bind=cores)
run_on() {
  local node="$1"; shift
  # Called as a background job so the tracked PID is the srun process itself.
  exec "${srun_base[@]}" --nodelist="$node" "${runtime[@]}" "$@"
}

steps=()
pending_step=""
token_created=0
cleanup() {
  local rc=$?
  trap - EXIT INT TERM USR1
  # The pending step or driver is stopped before the Ray nodes.
  if [[ -n "$pending_step" ]]; then steps+=("$pending_step"); fi
  for ((i=${#steps[@]}-1; i>=0; i--)); do
    kill "${steps[i]}" 2>/dev/null || true
  done
  # Leave time for native trainer supervisors to release their leases before
  # self-requeue, but bound teardown inside the four-minute warning window.
  deadline=$((SECONDS + 60))
  while (( SECONDS < deadline )); do
    alive=0
    for p in "${steps[@]}"; do kill -0 "$p" 2>/dev/null && alive=1 || true; done
    (( alive )) || break
    sleep 1
  done
  for p in "${steps[@]}"; do kill -KILL "$p" 2>/dev/null || true; done
  for p in "${steps[@]}"; do wait "$p" 2>/dev/null || true; done
  if (( token_created )); then rm -f -- "$RAY_AUTH_TOKEN_PATH"; fi
  if (( rc != 0 )); then
    echo "ERROR: Ray Tune allocation failed, exit $rc" >&2
    for f in "$ATTEMPT_ROOT"/*.log; do [[ -f "$f" ]] && tail -n 25 "$f" >&2; done
  fi
  echo "Ray Tune allocation exited=$rc, output=$RAY_TUNE_JOB_ROOT"
  pipeline_outcome "$rc"
  exit "$?"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'pipeline_interrupted=1; exit 75' USR1

allocation="$ATTEMPT_ROOT/allocation.tsv"
printf 'node\tcpus\tgpus\tcpu_affinity\tslurm_step_gpus\tcuda_visible_devices\n' > "$allocation"
CPUS_PER_NODE=""
GPUS_PER_NODE=""
for node in "${NODES[@]}"; do
  run_on "$node" "$python_exec" scripts/ray_tune_trials.py --slurm-node-resources > "$ATTEMPT_ROOT/node_resources.tmp" &
  pending_step="$!"
  wait "$pending_step"
  pending_step=""
  read -r cpus gpus _ < "$ATTEMPT_ROOT/node_resources.tmp"
  printf '%s\t%s\n' "$node" "$(<"$ATTEMPT_ROOT/node_resources.tmp")" >> "$allocation"
  [[ "$cpus" =~ ^[1-9][0-9]*$ && "$gpus" =~ ^[0-9]+$ ]] || { echo "Invalid resource probe for $node" >&2; exit 2; }
  if [[ -n "$CPUS_PER_NODE" && ( "$cpus" != "$CPUS_PER_NODE" || "$gpus" != "$GPUS_PER_NODE" ) ]]; then
    echo "Heterogeneous Slurm resources are unsupported; see $allocation" >&2
    exit 2
  fi
  CPUS_PER_NODE="$cpus"
  GPUS_PER_NODE="$gpus"
done
rm -f -- "$ATTEMPT_ROOT/node_resources.tmp"
if [[ "$NODE_COUNT" != "${RAY_TUNE_FIXED_NODES:-$NODE_COUNT}" ||
      "$CPUS_PER_NODE" != "${RAY_TUNE_FIXED_CPUS_PER_NODE:-$CPUS_PER_NODE}" ||
      "$GPUS_PER_NODE" != "${RAY_TUNE_FIXED_GPUS_PER_NODE:-$GPUS_PER_NODE}" ]]; then
  echo "Discovered allocation differs from the fixed workload layout; see $allocation" >&2
  exit 2
fi
if [[ "${PIPELINE_STAGE_COMPLETE:-0}" == 1 ]]; then pipeline_action finish; exit 0; fi
export RAY_EXPECTED_NODES="$NODE_COUNT" RAY_EXPECTED_GPUS="$(( NODE_COUNT * GPUS_PER_NODE ))"
export RAY_EXPECTED_CPUS_PER_NODE="$CPUS_PER_NODE" RAY_EXPECTED_GPUS_PER_NODE="$GPUS_PER_NODE"
echo "Git=$(git rev-parse HEAD) job=$SLURM_JOB_ID partition=${SLURM_JOB_PARTITION:-unknown} account=${SLURM_JOB_ACCOUNT:-unknown} qos=${SLURM_JOB_QOS:-${SBATCH_QOS:-unknown}}"
echo "Requested CPUs/task=$REQUESTED_CPUS allocated CPUs/node=${SLURM_JOB_CPUS_PER_NODE:-unknown}"
echo "Experiment=$EXPERIMENT_ID sweep=$SWEEP_SPEC outputs=$RAY_TUNE_JOB_ROOT"
cat "$allocation"

# Reject invalid architecture/source matrices before starting any Ray workers.
# Validation uses temporary CPU models and never writes the saved experiment.
if [[ "$PREFLIGHT_ONLY" == 1 || ( "$RESTORE" == 0 && -n "$SWEEP_SPEC" ) ]]; then
  run_on "$HEAD" "$python_exec" scripts/ray_tune_trials.py "$@" --preflight-only \
    --cluster-nodes "$NODE_COUNT" --cpus-per-node "$CPUS_PER_NODE" --gpus-per-node "$GPUS_PER_NODE" &
  pending_step="$!"
  wait "$pending_step"
  pending_step=""
  if [[ "$PREFLIGHT_ONLY" == 1 ]]; then exit 0; fi
fi

[[ ! -e "$RAY_AUTH_TOKEN_PATH" ]] || { echo "Ray authentication token already exists" >&2; exit 2; }
token_created=1
"${runtime[@]}" "$python_exec" -c 'import secrets; print(secrets.token_urlsafe(48))' > "$RAY_AUTH_TOKEN_PATH"
[[ -s "$RAY_AUTH_TOKEN_PATH" ]] || exit 2
common_args=(--num-cpus="$CPUS_PER_NODE" --num-gpus="$GPUS_PER_NODE"
  "--node-manager-port=$NODE_PORT" "--object-manager-port=$OBJECT_PORT"
  "--metrics-export-port=$METRICS_PORT" "--runtime-env-agent-port=$RUNTIME_PORT"
  "--dashboard-agent-listen-port=$AGENT_HTTP_PORT" "--dashboard-agent-grpc-port=$AGENT_GRPC_PORT"
  "--min-worker-port=$MIN_WORKER_PORT" "--max-worker-port=$MAX_WORKER_PORT"
  --disable-usage-stats --block)
if [[ "$ENABLE_DASHBOARD" == 1 ]]; then
  echo "Dashboard on $HEAD at http://127.0.0.1:$DASHBOARD_PORT; token file: $RAY_AUTH_TOKEN_PATH"
fi
run_on "$HEAD" "$ray_exec" start \
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
  run_on "$node" "$ray_exec" start \
    --address="$RAY_ADDRESS" --node-ip-address="$ip" --temp-dir="$TEMP_DIR" \
    "${common_args[@]}" > "$ATTEMPT_ROOT/worker_${i}.log" 2>&1 &
  steps+=("$!")
done

# Optional low-frequency GPU utilization samples; results live alongside Ray logs.
# Missing nvidia-smi is nonfatal; native perf/* counters still remain available.
if [[ "$PROFILE_GPU_MONITOR" == 1 && "$GPUS_PER_NODE" != 0 ]]; then
  for ((i=0; i<NODE_COUNT; i++)); do
    run_on "${NODES[i]}" nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used \
      --format=csv --loop-ms=5000 > "$ATTEMPT_ROOT/gpu_${i}.csv" 2>&1 &
    steps+=("$!")
  done
fi

run_on "$HEAD" "$python_exec" -u scripts/ray_tune_trials.py "$@" \
  > "$ATTEMPT_ROOT/driver.log" 2>&1 &
pending_step="$!"
echo "Driver log: $ATTEMPT_ROOT/driver.log"
wait "$pending_step"
pending_step=""
if [[ -n "${PIPELINE_ROOT:-}" ]]; then pipeline_action finish; fi
if (( 10#$HOLD_SECONDS > 0 )); then
  sleep "$(( 10#$HOLD_SECONDS ))" &
  pending_step="$!"
  wait "$pending_step"
  pending_step=""
fi
