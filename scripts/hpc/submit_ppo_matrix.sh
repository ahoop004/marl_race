#!/usr/bin/env bash
# Run with: bash scripts/hpc/submit_ppo_matrix.sh --dry-run
#           bash scripts/hpc/submit_ppo_matrix.sh --submit
# Jobs are dependency-chained: at most one matrix allocation runs at once.
set -Eeuo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
MODE=dry-run
PROFILE=""
SUITE=full
TAG="$(date -u +%Y%m%dT%H%M%SZ)"
STEPS=262144
WALLTIME=02:00:00
while (($#)); do
  case "$1" in
    --submit) MODE=submit; shift;;
    --dry-run) MODE=dry-run; shift;;
    --profile) PROFILE="${2:?}"; shift 2;;
    --suite) SUITE="${2:?}"; shift 2;;
    --tag) TAG="${2:?}"; shift 2;;
    --steps) STEPS="${2:?}"; shift 2;;
    --walltime) WALLTIME="${2:?}"; shift 2;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
[[ "$SUITE" == full || "$SUITE" == nodes || "$SUITE" == workers ]] || { echo "--suite full|nodes|workers" >&2; exit 2; }
[[ "$TAG" =~ ^[A-Za-z0-9_-]+$ && "$STEPS" =~ ^[0-9]+$ ]] || exit 2
(( STEPS >= 131072 && STEPS % 32768 == 0 )) || { echo "--steps must be >=131072 and divisible by 32768" >&2; exit 2; }
[[ "$WALLTIME" =~ ^[0-9]{2}:[0-5][0-9]:[0-5][0-9]$ ]] || { echo "--walltime must be HH:MM:SS" >&2; exit 2; }
[[ -f scripts/hpc/ray_tune_matrix.sbatch ]] || { echo "Missing generalized matrix launcher" >&2; exit 2; }
if [[ "$MODE" == submit ]]; then
  command -v sbatch >/dev/null || { echo "sbatch unavailable" >&2; exit 2; }
  [[ "$(git branch --show-current)" == multiNode ]] || { echo "Expected multiNode" >&2; exit 2; }
  [[ -z "$(git status --porcelain -- src configs scenarios scripts run.py)" ]] || {
    echo "Commit/review source changes before launching (T5 revision guard)" >&2; exit 2;
  }
fi

CRUN="${CRUN:-$(command -v crun || true)}"
[[ -n "$CRUN" ]] || CRUN=/cm/shared/containers/python3/2025.1-py312/crun.python3
VENV="${VENV:-$HOME/envs/marl-race}"
[[ -x "$CRUN" && -d "$VENV" ]] || { echo "Missing project container/environment: $CRUN -p $VENV" >&2; exit 2; }

OUT="${PROFILE_MATRIX_ROOT:-$REPO/outputs/hpc_profile_matrix}/$TAG"
mkdir -p "$(dirname "$OUT")"
mkdir "$OUT" || { echo "Matrix tag already exists; choose a new --tag to preserve previous outputs" >&2; exit 2; }
SPEC_DIR="$OUT/specs"
mkdir -p "$SPEC_DIR"
JOBS="$OUT/jobs.tsv"
printf 'profile\texperiment_id\tjob_id\tnodes\tgpus_per_node\tcpus_per_node\tworkers\tenvs_per_worker\ttrials\tsteps_per_trial\tdependency\n' > "$JOBS"

# profile, nodes, GPUs/node, CPUs/node, collector processes/trial, envs/worker, Ray CPUs/trial
profiles=(
  'w04_n1g1 1 1 24 4 8 6'
  'w08_n1g1 1 1 24 8 8 10'
  'w16_n1g1 1 1 24 16 8 20'
  'n02_g1   2 1 24 16 8 20'
  'n03_g1   3 1 24 16 8 20'
  'n02_g2   2 2 24 8 8 10'
  'n03_g2   3 2 24 8 8 10'
)
selected=()
for record in "${profiles[@]}"; do
  read -r key nodes gpus cpus workers envs ray_cpus <<< "$record"
  if [[ "$SUITE" == workers && "$nodes" != 1 ]] || \
     [[ "$SUITE" == nodes && "$key" == w04_n1g1 || "$SUITE" == nodes && "$key" == w08_n1g1 ]]; then
    continue
  fi
  [[ -z "$PROFILE" || "$PROFILE" == "$key" ]] || continue
  selected+=("$record")
  concurrency=$(( nodes * gpus ))
  experiment_id="profile_${TAG}_${key}"
  [[ ! -e "$REPO/outputs/ray_tune_${experiment_id}" ]] || { echo "Existing experiment $experiment_id; choose a new --tag" >&2; exit 2; }
  spec="$SPEC_DIR/${key}.yaml"
  cat > "$spec" <<YAML
# Throughput comparison only. Fixed evaluation is deliberately short and not a model-quality test.
name: PPO HPC resource profile ${key}
experiment_id: ${experiment_id}
algorithm: ppo
scenario: scenarios/ppo_lap_completion_pretrain.yaml
total_steps: ${STEPS}
seeds: [42, 43, 44, 45, 46, 47]
sweep_seed: 2026
search:
  method: grid
  parameters:
    algorithm.params.learning_rate:
      type: choice
      values: [0.001]
max_concurrent_trials: ${concurrency}
resources_per_trial: {cpu: ${ray_cpus}, gpu: 1}
fixed_overrides:
  - experiment.parallel_envs=${workers}
  - experiment.envs_per_worker=${envs}
  - experiment.measure_stages=true
  - experiment.measure_services=true
  - algorithm.params.n_steps=32768
  - algorithm.params.frames_per_batch=4096
  - algorithm.params.batch_size=1024
  - algorithm.params.n_epochs=10
evaluation:
  seed: 10042
  episodes: 1
  every_steps: ${STEPS}
  max_steps: 256
wandb:
  mode: disabled
YAML
done
(( ${#selected[@]} > 0 )) || { echo "No profiles selected (check --profile and --suite)" >&2; exit 2; }

# Validate the entire selection before the first sbatch call, including dry runs.
mkdir -p "$OUT/validation"
for record in "${selected[@]}"; do
  read -r key nodes gpus cpus workers envs ray_cpus <<< "$record"
  echo "Validating $key: nodes=$nodes GPUs/node=$gpus CPUs/node=$cpus workers/trial=$workers envs/worker=$envs"
  if ! RESTORE=0 "$CRUN" -p "$VENV" python3 scripts/ray_tune_trials.py \
      --preflight-only --sweep-spec "$SPEC_DIR/$key.yaml" --experiment-id "profile_${TAG}_${key}" \
      --cluster-nodes "$nodes" --cpus-per-node "$cpus" --gpus-per-node "$gpus" \
      > "$OUT/validation/$key.log" 2>&1; then
    echo "PREFLIGHT FAILED: $key; zero jobs submitted" >&2
    cat "$OUT/validation/$key.log" >&2
    exit 1
  fi
  cat "$OUT/validation/$key.log"
done

previous=""
for record in "${selected[@]}"; do
  read -r key nodes gpus cpus workers envs ray_cpus <<< "$record"
  experiment_id="profile_${TAG}_${key}"
  spec="$SPEC_DIR/${key}.yaml"
  dependency="none"
  command=(sbatch --parsable --partition=gpu --account=q1wang --qos=normal
    --nodes="$nodes" --ntasks="$nodes" --ntasks-per-node=1
    --cpus-per-task="$cpus" --gres="gpu:${gpus}" --time="$WALLTIME")
  if [[ -n "$previous" ]]; then
    dependency="afterok:$previous"
    command+=("--dependency=$dependency" --kill-on-invalid-dep=no)
  fi
  command+=("--export=ALL,RESTORE=0,EXPERIMENT_ID=${experiment_id},SWEEP_SPEC=${spec},PROFILE_GPUS_PER_NODE=${gpus},RAY_TUNE_JOB_ROOT=${REPO}/outputs/ray_tune_${experiment_id}"
            scripts/hpc/ray_tune_matrix.sbatch)
  if [[ "$MODE" == submit ]]; then
    if ! job="$("${command[@]}")"; then
      echo "Submission failed for $key; submitted IDs are preserved in $JOBS" >&2
      printf 'Cancel recorded pending jobs: tail -n +2 %q | cut -f3 | xargs -r scancel --state=PENDING\n' "$JOBS" >&2
      exit 1
    fi
    job="${job%%;*}"
    [[ "$job" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response $job; earlier IDs are in $JOBS" >&2; exit 1; }
    previous="$job"
    echo "$key: submitted $job; dependency=$dependency"
  else
    job=DRY_RUN
    previous="<${key}_job_id>"
    echo "$key: validated dry run; dependency=$dependency"
    printf '  %q' "${command[@]}"; printf '\n'
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t6\t%s\t%s\n' \
    "$key" "$experiment_id" "$job" "$nodes" "$gpus" "$cpus" "$workers" "$envs" "$STEPS" "$dependency" >> "$JOBS"
done
printf 'Matrix directory: %s\nManifest: %s\n' "$OUT" "$JOBS"
if [[ "$MODE" == dry-run ]]; then
  echo 'All selected profiles passed preflight. No Slurm jobs were submitted; dependency IDs above are placeholders.'
else
  echo 'Chain: each profile requires afterok of the previous job. Failed dependencies remain pending (DependencyNeverSatisfied).'
  printf 'Inspect: tail -n +2 %q | cut -f3 | paste -sd, - | xargs -r squeue -o "%%.18i %%.12T %%.35E %%R" --jobs\n' "$JOBS"
  printf 'Cancel remaining: tail -n +2 %q | cut -f3 | xargs -r scancel --state=PENDING\n' "$JOBS"
fi
