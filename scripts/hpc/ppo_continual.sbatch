#!/usr/bin/env bash
#SBATCH --job-name=ppo-continual
#SBATCH --partition=reserved-q1wang
#SBATCH --account=q1wang
#SBATCH --qos=q1wang
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=4-00:00:00
#SBATCH --output=ppo-continual-%j.out
#SBATCH --error=ppo-continual-%j.err
# Compatibility preset; all runtime orchestration is shared.
SCENARIO=${SCENARIO:-scenarios/ppo_continual.yaml}
source "${REPO_DIR:-$HOME/marl_race}/scripts/hpc/launch_single_node_gpu.sh" "$@"
