"""Compatibility entry point for independent PPO LoRA family experiments."""
from launch_experiment import main


if __name__ == "__main__":
    main(default_scenario="scenarios/ppo_generated_lora_20k.yaml")
