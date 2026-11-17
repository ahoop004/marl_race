#!/usr/bin/env bash
#SBATCH --job-name=race-experiment
#SBATCH --partition=reserved-q1wang
#SBATCH --account=q1wang
#SBATCH --qos=q1wang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=4-00:00:00
#SBATCH --output=race-experiment-%j.out
#SBATCH --error=race-experiment-%j.err
# Override hardware with sbatch options; pass Python options after this filename.
SCENARIO=${SCENARIO:-scenarios/ppo_generated_lora_20k.yaml}
source "${REPO_DIR:-$HOME/marl_race}/scripts/hpc/launch_single_node_gpu.sh" "$@"
