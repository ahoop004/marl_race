#!/usr/bin/env bash
#SBATCH --job-name=marl-ray-smoke
#SBATCH --partition=gpu
#SBATCH --account=q1wang
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=ray-2node-%j.log

set -Eeuo pipefail

# Submit from ~/marl_race:
#   sbatch scripts/hpc/ray_2node_smoke.sbatch
# Optional dashboard (only on the head node, bound to loopback):
#   sbatch --export=ALL,ENABLE_DASHBOARD=1,HOLD_SECONDS=300 scripts/hpc/ray_2node_smoke.sbatch

REPO="$HOME/marl_race"
VENV="$HOME/envs/marl-race"
ENABLE_DASHBOARD="${ENABLE_DASHBOARD:-0}"
HOLD_SECONDS="${HOLD_SECONDS:-0}"

cd "$REPO"
[[ -d "$VENV" ]] || { echo "Missing environment: $VENV" >&2; exit 1; }
[[ "$ENABLE_DASHBOARD" == 0 || "$ENABLE_DASHBOARD" == 1 ]] || {
  echo "ENABLE_DASHBOARD must be 0 or 1" >&2; exit 1;
}
[[ "$HOLD_SECONDS" =~ ^[0-9]+$ ]] || { echo "HOLD_SECONDS must be an integer" >&2; exit 1; }
(( HOLD_SECONDS <= 1800 )) || { echo "HOLD_SECONDS must be <= 1800" >&2; exit 1; }

# The container launcher may not be on PATH in a non-interactive Slurm shell.
CRUN="$(command -v crun || true)"
if [[ -z "$CRUN" ]]; then
  CRUN=/cm/shared/containers/python3/2025.1-py312/crun.python3
fi
[[ -x "$CRUN" ]] || { echo "Cannot find crun: $CRUN" >&2; exit 1; }

export PYTHONUNBUFFERED=1
export RAY_DISABLE_USAGE_STATS=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
OUT="$REPO/outputs/ray_smoke_${SLURM_JOB_ID}"
mkdir -p "$OUT"
chmod 700 "$OUT"

# All Ray processes use one job-specific shared token file.  Wahab's home
# filesystem is visible from both compute nodes.  Never print this token.
umask 077
export RAY_AUTH_MODE=token
unset RAY_AUTH_TOKEN || true
export RAY_AUTH_TOKEN_PATH="$OUT/.ray_auth_token"
[[ ! -e "$RAY_AUTH_TOKEN_PATH" ]] || {
  echo "Refusing to overwrite an existing Ray authentication token" >&2
  exit 1
}
# Also clean up if startup fails before the full cleanup handler is installed.
trap 'rm -f -- "$RAY_AUTH_TOKEN_PATH"' EXIT
"$CRUN" -p "$VENV" python3 -c 'import secrets; print(secrets.token_urlsafe(48))' > "$RAY_AUTH_TOKEN_PATH"
chmod 600 "$RAY_AUTH_TOKEN_PATH"
[[ -s "$RAY_AUTH_TOKEN_PATH" ]] || { echo "Failed to create auth token" >&2; exit 1; }

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#NODES[@]} -eq 2 && ${NODES[0]} != "${NODES[1]}" ]] || {
  echo "Expected two distinct nodes, got: ${NODES[*]}" >&2; exit 1;
}
HEAD="${NODES[0]}"
WORKER="${NODES[1]}"

node_ip() { getent ahostsv4 "$1" | awk '$2=="STREAM" {print $1; exit}'; }
HEAD_IP="$(node_ip "$HEAD")"
WORKER_IP="$(node_ip "$WORKER")"
[[ -n "$HEAD_IP" && -n "$WORKER_IP" ]] || {
  echo "Failed to resolve compute-node IPv4 addresses" >&2; exit 1;
}

# Reserve one job-specific port block to avoid ordinary multi-user collisions.
# Modulo collisions are still possible; head/worker logs will show bind errors.
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
HEAD_ADDR="$HEAD_IP:$GCS_PORT"
export RAY_ADDRESS="$HEAD_ADDR"
export RAY_SMOKE_RESULTS="$OUT/results.json"
export EXPECTED_RAY_HOSTS="$HEAD,$WORKER"

head_step=""
worker_step=""
cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  [[ -z "$worker_step" ]] || kill "$worker_step" 2>/dev/null || true
  [[ -z "$head_step" ]] || kill "$head_step" 2>/dev/null || true
  [[ -z "$worker_step" ]] || wait "$worker_step" 2>/dev/null || true
  [[ -z "$head_step" ]] || wait "$head_step" 2>/dev/null || true
  # Remove the secret after the Ray steps have been stopped, on success or failure.
  rm -f -- "$RAY_AUTH_TOKEN_PATH"
  if (( rc != 0 )); then
    echo "FAILED (exit $rc): recent logs follow" >&2
    for f in "$OUT/head.log" "$OUT/worker.log" "$OUT/driver.log"; do
      if [[ -f "$f" ]]; then
        echo "--- $f ---" >&2
        tail -n 50 "$f" >&2 || true
      fi
    done
  fi
  echo "Ray validation finished, exit=$rc; results=$OUT"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'Repository: %s\nEnvironment: %s\nContainer: %s\n' "$REPO" "$VENV" "$CRUN"
printf 'Head: %s (%s)\nWorker: %s (%s)\nRay GCS: %s\n' "$HEAD" "$HEAD_IP" "$WORKER" "$WORKER_IP" "$HEAD_ADDR"
printf 'Dashboard enabled: %s; hold: %s seconds\n' "$ENABLE_DASHBOARD" "$HOLD_SECONDS"
printf 'Ray authentication: token (job-specific shared file, deleted on exit)\n'
printf 'Logs and results: %s\n' "$OUT"

# Every srun step is pinned to one of the two allocated nodes. --overlap
# permits the validation driver to run alongside the persistent head step.
srun_base=(srun --overlap --nodes=1 --ntasks=1 --ntasks-per-node=1
           --cpus-per-task=8 --gres=gpu:1)
run_on() {
  local node="$1"
  shift
  "${srun_base[@]}" --nodelist="$node" "$CRUN" -p "$VENV" "$@"
}

# Ray advertises only the CPUs/GPUs Slurm allocated, not the whole machine.
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

# /tmp is node-local on typical Slurm installations and avoids shared FS
# overhead for Ray's socket/session files. Use a unique path per Slurm job.
TEMP_DIR="/tmp/ray_${USER}_${SLURM_JOB_ID}"

"${srun_base[@]}" --nodelist="$HEAD" "$CRUN" -p "$VENV" \
  ray start --head --node-ip-address="$HEAD_IP" --port="$GCS_PORT" \
  --ray-client-server-port="$CLIENT_PORT" \
  --include-dashboard="$([[ "$ENABLE_DASHBOARD" == 1 ]] && echo true || echo false)" \
  --dashboard-host=127.0.0.1 --dashboard-port="$DASHBOARD_PORT" \
  --temp-dir="$TEMP_DIR" \
  "${common_args[@]}" > "$OUT/head.log" 2>&1 &
head_step=$!

# Wait for the head GCS to accept TCP connections.
ready=0
for ((i=0; i<90; i++)); do
  if (echo >/dev/tcp/"$HEAD_IP"/"$GCS_PORT") >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "$head_step" 2>/dev/null; then
    echo "Head Slurm step exited during startup" >&2
    exit 1
  fi
  sleep 2
done
(( ready == 1 )) || { echo "Timed out waiting for Ray head" >&2; exit 1; }

"${srun_base[@]}" --nodelist="$WORKER" "$CRUN" -p "$VENV" \
  ray start --address="$HEAD_ADDR" --node-ip-address="$WORKER_IP" \
  --temp-dir="$TEMP_DIR" \
  "${common_args[@]}" > "$OUT/worker.log" 2>&1 &
worker_step=$!

# Write a small driver in the shared outputs directory and execute it from
# the head node in the same container and Python environment as Ray itself.
cat > "$OUT/check_ray_cluster.py" <<'PY'
import json
import os
import socket
import time
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

assert os.environ.get("RAY_AUTH_MODE") == "token", "Ray token auth not enabled"
auth_file = Path(os.environ["RAY_AUTH_TOKEN_PATH"])
assert auth_file.is_file() and auth_file.stat().st_size > 0, "Ray auth token missing"
expected = set(os.environ["EXPECTED_RAY_HOSTS"].split(","))
address = os.environ["RAY_ADDRESS"]
for attempt in range(30):
    try:
        ray.init(address=address, logging_level="ERROR")
        break
    except Exception:
        if attempt == 29:
            raise
        time.sleep(2)

try:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        nodes = [n for n in ray.nodes() if n["Alive"]]
        resources = ray.cluster_resources()
        if len(nodes) == 2 and resources.get("GPU", 0) >= 2:
            break
        time.sleep(2)
    else:
        raise RuntimeError(f"Cluster not ready: {ray.nodes()}; {ray.cluster_resources()}")

    @ray.remote(num_cpus=1, num_gpus=1)
    def gpu_check():
        import torch
        gpu_ids = [str(x) for x in ray.get_gpu_ids()]
        available = torch.cuda.is_available()
        value = float(torch.ones(1, device="cuda").item()) if available else None
        return {
            "hostname": socket.gethostname().split(".")[0],
            "ray_gpu_ids": gpu_ids,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_cuda_available": available,
            "torch_cuda_device_count": torch.cuda.device_count(),
            "gpu_name": torch.cuda.get_device_name(0) if available else None,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "gpu_compute_capability": list(torch.cuda.get_device_capability(0)) if available else None,
            "cuda_test_value": value,
        }

    tasks = [
        gpu_check.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
            node_id=n["NodeID"], soft=False)).remote()
        for n in nodes
    ]
    results = ray.get(tasks, timeout=120)
    actual_hosts = {r["hostname"] for r in results}
    assert actual_hosts == expected, (actual_hosts, expected)
    assert all(r["torch_cuda_available"] and r["torch_cuda_device_count"] == 1
               and len(r["ray_gpu_ids"]) == 1 and r["cuda_test_value"] == 1.0
               for r in results), results

    report = {
        "status": "PASS",
        "ray_version": ray.__version__,
        "authentication": "token",
        "address": address,
        "resources": ray.cluster_resources(),
        "nodes": [{"name": n["NodeManagerHostname"], "id": n["NodeID"]} for n in nodes],
        "gpu_tasks": results,
    }
    Path(os.environ["RAY_SMOKE_RESULTS"]).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print("ALL RAY MULTI-NODE SMOKE CHECKS PASSED", flush=True)
finally:
    ray.shutdown()
PY

run_on "$HEAD" python3 -u "$OUT/check_ray_cluster.py" 2>&1 | tee "$OUT/driver.log"

if [[ "$ENABLE_DASHBOARD" == 1 ]]; then
  echo "Dashboard on head node ONLY: http://127.0.0.1:$DASHBOARD_PORT"
  echo "Head node: $HEAD (SSH tunneling requires compute-node SSH permission)"
  echo "Port forwarding must terminate on the head node's loopback interface."
fi
if (( HOLD_SECONDS > 0 )); then
  echo "Holding Ray cluster for $HOLD_SECONDS seconds for inspection..."
  sleep "$HOLD_SECONDS"
fi