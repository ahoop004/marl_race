#!/usr/bin/env bash
#SBATCH --job-name=ppo-lora-cpu
#SBATCH --partition=main
#SBATCH --nodes=5
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --output=ppo-lora-cpu-%j.out
#SBATCH --error=ppo-lora-cpu-%j.err

# Account, QoS, memory and time come from the main partition or sbatch flags.
CPU_INDEPENDENT=1
SCENARIO=${SCENARIO:-scenarios/ppo_lora_families.yaml}
source "${REPO_DIR:-$HOME/marl_race}/scripts/hpc/launch_single_node_gpu.sh" "$@"
