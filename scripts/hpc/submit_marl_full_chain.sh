#!/usr/bin/env bash
# Queue checkpointed full PPO and MAPPO experiments: PPO -> (LoRA r4 || full fine-tune).
# Example: bash submit_marl_full_chain.sh
# Dry run: DRY_RUN=1 bash submit_marl_full_chain.sh
set -euo pipefail
REPO_DIR=${REPO_DIR:-"$HOME/marl_race"}
cd "$REPO_DIR"
[[ "$(git branch --show-current)" == multiGPU ]] || { echo "Expected multiGPU branch" >&2; exit 2; }
EXPECTED_SHA=$(git rev-parse HEAD)
# Pin the known configuration revision; edit deliberately for later revisions.
PINNED_SHA=8e325d0bb04bc9f56107b7420a0cbb84f42baff5
[[ "$EXPECTED_SHA" == "$PINNED_SHA" ]] || { echo "Expected $PINNED_SHA, got $EXPECTED_SHA" >&2; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo "Commit/stash tracked changes before launching" >&2; exit 2; }

SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/marl_full_chain.sbatch"
[[ -f "$SCRIPT" ]] || { echo "Missing $SCRIPT" >&2; exit 2; }
TOTAL=120000000
# Segment boundaries are whole PPO-rollout multiples, except the final remainder.
PPO_CHUNK=${PPO_CHUNK:-12288000}    # 30 updates per segment, ~10 jobs.
MAPPO_CHUNK=${MAPPO_CHUNK:-4096000}  # 10 updates per segment, ~30 jobs/arm.
for step in "$PPO_CHUNK" "$MAPPO_CHUNK"; do
  (( step > 0 && step % 409600 == 0 )) || { echo "Chunk must be positive multiple of 409600" >&2; exit 2; }
done
RUN_ROOT=${RUN_ROOT:-"$REPO_DIR/outputs/training/marl_full_ppo_mappo_${EXPECTED_SHA:0:7}_s42"}
RUN_ROOT=$(realpath -m "$RUN_ROOT")
PPO_CPUS=${PPO_CPUS:-24}; PPO_WORKERS=${PPO_WORKERS:-16}
MAPPO_CPUS=${MAPPO_CPUS:-48}; MAPPO_WORKERS=${MAPPO_WORKERS:-32}
ENVS_PER_WORKER=${ENVS_PER_WORKER:-8}
PPO_MEM=${PPO_MEM:-128G}; MAPPO_MEM=${MAPPO_MEM:-192G}
PPO_WALLTIME=${PPO_WALLTIME:-08:00:00}
MAPPO_WALLTIME=${MAPPO_WALLTIME:-12:00:00}
DRY_RUN=${DRY_RUN:-0}

# Verify Slurm queue/QoS walltime and job-count limits before submitting 70 jobs.
printf 'Repository: %s\nSHA: %s\nOutputs: %s\n' "$REPO_DIR" "$EXPECTED_SHA" "$RUN_ROOT"
printf 'PPO: %s workers, %s cpus, %s memory, %s/job; MAPPO: %s workers, %s cpus, %s memory, %s/job\n' \
  "$PPO_WORKERS" "$PPO_CPUS" "$PPO_MEM" "$PPO_WALLTIME" \
  "$MAPPO_WORKERS" "$MAPPO_CPUS" "$MAPPO_MEM" "$MAPPO_WALLTIME"

jobs=0
submit_one() {
  local phase="$1" target="$2" previous="$3"
  local workers cpus mem walltime
  if [[ "$phase" == ppo ]]; then
    workers=$PPO_WORKERS cpus=$PPO_CPUS mem=$PPO_MEM walltime=$PPO_WALLTIME
  else
    workers=$MAPPO_WORKERS cpus=$MAPPO_CPUS mem=$MAPPO_MEM walltime=$MAPPO_WALLTIME
  fi
  local opts=(--parsable --job-name="marl-${phase}-${target}"
    --cpus-per-task="$cpus" --mem="$mem" --time="$walltime")
  if [[ -n "$previous" ]]; then
    opts+=(--dependency="afterok:$previous")
  fi
  local exports="ALL,PHASE=$phase,STOP_AT=$target,RUN_ROOT=$RUN_ROOT,EXPECTED_SHA=$EXPECTED_SHA,WORKERS=$workers,ENVS_PER_WORKER=$ENVS_PER_WORKER,REPO_DIR=$REPO_DIR"
  opts+=(--export="$exports")
  local job_id
  if (( DRY_RUN )); then
    job_id=$((100000 + jobs))
    printf 'DRY-RUN job=%s phase=%s stop=%s depends_on=%s cpus=%s workers=%s\n' \
      "$job_id" "$phase" "$target" "${previous:-none}" "$cpus" "$workers" >&2
  else
    job_id=$(sbatch "${opts[@]}" "$SCRIPT")
    job_id=${job_id%%;*}
    [[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch job id: $job_id" >&2; exit 3; }
    printf 'Queued %-8s stop=%9s job=%s after=%s\n' "$phase" "$target" "$job_id" "${previous:-none}" >&2
  fi
  jobs=$((jobs+1))
  REPLY="$job_id"
}
submit_phase() {
  local phase="$1" predecessor="$2" chunk="$3"
  local target=$chunk previous="$predecessor"
  while (( target < TOTAL )); do
    submit_one "$phase" "$target" "$previous"
    previous="$REPLY"
    target=$((target + chunk))
  done
  submit_one "$phase" "$TOTAL" "$previous"
}

# No job changes source checkpoint after PPO finishes. Both MAPPO arms depend
# on the final PPO segment but are independent of each other.
submit_phase ppo '' "$PPO_CHUNK"
ppo_last="$REPLY"
submit_phase lora "$ppo_last" "$MAPPO_CHUNK"
lora_last="$REPLY"
submit_phase finetune "$ppo_last" "$MAPPO_CHUNK"
finetune_last="$REPLY"
cat <<MSG
Queued completion chain (DRY_RUN=$DRY_RUN):
  PPO final job:       $ppo_last
  LoRA final job:      $lora_last
  Fine-tune final job: $finetune_last
  Output root: $RUN_ROOT
  All runs use the same full-budget PPO source best.pt for MAPPO initialization.
MSG
