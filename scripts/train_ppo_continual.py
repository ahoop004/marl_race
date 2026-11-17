"""Compatibility entry point for sequential continual PPO experiments."""
from launch_experiment import main


if __name__ == "__main__":
    main(default_scenario="scenarios/ppo_continual.yaml")
