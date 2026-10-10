"""CLI parsing and effective scenario validation."""
import argparse
import sys
from pathlib import Path
from core.scenario import (ScenarioError, apply_parameter_overrides,
                           load_and_expand_scenario, resolve_max_speed)
from application.configuration import validate_experiment_scenario
from core.agent_roles import resolve_agent_roles
from application.checkpoints import resolve_scenario_relative_path
from loggers.console import ConsoleLogger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="F110 RL training")
    p.add_argument("--scenario", required=True, help="Path to scenario YAML file")
    p.add_argument("--set", dest="parameter_overrides", action="append", default=[], metavar="KEY=YAML",
                   help="Repeatable scenario parameter override, e.g. training_defaults.lora={mode: shared, rank: 4}; "
                        "use !delete to remove an optional key. Dedicated CLI flags take precedence.")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--render", action="store_true")
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-speed", type=float, default=None,
                   help="Positive finite shared forward speed limit in m/s for learner vehicles and MPCs; "
                        "overrides environment.max_speed and individual forward limits")
    budget_args = p.add_mutually_exclusive_group()
    budget_args.add_argument("--episodes", type=int, default=None)
    budget_args.add_argument("--total-steps", type=int, default=None,
                             help="Aggregate environment-decision budget for PPO/MAPPO")
    p.add_argument("--max-steps", type=int, default=None,
                   help="Positive per-episode physics-step limit for bounded testing; also caps evaluation")
    p.add_argument("--torch-threads", type=int, default=None,
                   help="PyTorch CPU threads for training and evaluation")
    p.add_argument("--eval", action="store_true", help="Run evaluation instead of training")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Checkpoint file or run directory (best_model.pt): evaluate with --eval, "
                        "or initialize PPO training weights with a fresh optimizer; "
                        "overrides experiment.checkpoint in the scenario")
    p.add_argument("--pretrained-actor", type=str, default=None,
                   help="PPO or plain shared-MAPPO checkpoint/run directory to initialize actors; fresh critic and optimizer")
    p.add_argument(
        "--allow-provenance-mismatch",
        action="store_true",
        help="Allow --eval with a checkpoint from a different scenario/config/map contract.",
    )
    p.add_argument("--eval-episodes", type=int, default=None, help="Evaluation episodes; defaults to --episodes")
    p.add_argument("--eval-protocol", choices=("selection", "final"), default=None,
                   help="Use fixed scenario evaluation seeds, episodes, and horizon; requires --eval.")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def apply_cli_overrides(scenario: dict, args: argparse.Namespace) -> dict:
    if getattr(args, "parameter_overrides", None):
        scenario = apply_parameter_overrides(scenario, args.parameter_overrides)
    if getattr(args, "max_speed", None) is not None:
        scenario.setdefault("environment", {})["max_speed"] = args.max_speed
    if args.seed is not None:
        scenario.setdefault("experiment", {})["seed"] = args.seed
    if args.episodes is not None:
        scenario.setdefault("experiment", {})["episodes"] = args.episodes
        scenario["experiment"].pop("total_steps", None)
    if getattr(args, "total_steps", None) is not None:
        scenario.setdefault("experiment", {})["total_steps"] = args.total_steps
    max_steps = getattr(args, "max_steps", None)
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("--max-steps must be positive")
        scenario.setdefault("environment", {})["max_steps"] = max_steps
        if scenario.get("evaluation"):
            scenario["evaluation"]["max_steps"] = max_steps
    if args.wandb:
        scenario.setdefault("wandb", {})["enabled"] = True
    elif args.no_wandb:
        scenario.setdefault("wandb", {})["enabled"] = False
    if args.render:
        scenario.setdefault("environment", {})["render"] = True
    elif args.no_render:
        scenario.setdefault("environment", {})["render"] = False
    torch_threads = getattr(args, "torch_threads", None)
    if torch_threads is not None:
        scenario.setdefault("experiment", {})["torch_threads"] = torch_threads
    return resolve_max_speed(scenario)


def main() -> None:
    args = parse_args()
    console = ConsoleLogger(verbose=not args.quiet)
    if args.eval_protocol and (not args.eval or args.eval_episodes is not None):
        raise ValueError("--eval-protocol requires --eval and uses fixed episodes; omit --eval-episodes.")

    try:
        # Apply speed choices before expansion so a CLI override can replace or
        # remove a YAML speed limit before it changes the physical/controller limits.
        load_overrides = list(args.parameter_overrides)
        if args.max_speed is not None:
            load_overrides.append(f"environment.max_speed={args.max_speed}")
        scenario = load_and_expand_scenario(args.scenario, validate=False, overrides=load_overrides)
    except (ScenarioError, FileNotFoundError) as exc:
        console.print_error(f"Failed to load scenario: {exc}")
        sys.exit(1)

    try:
        scenario = apply_cli_overrides(scenario, args)
        validate_experiment_scenario(scenario)
    except ScenarioError as exc:
        console.print_error(f"Invalid scenario after CLI overrides: {exc}")
        sys.exit(1)
    if scenario["experiment"].get("torch_threads") is not None:
        import torch
        torch.set_num_threads(scenario["experiment"]["torch_threads"])
    scenario_dir = Path(args.scenario).resolve().parent
    if args.checkpoint is None:
        configured_checkpoint = scenario["experiment"].get("checkpoint")
        if configured_checkpoint is not None:
            args.checkpoint = str(resolve_scenario_relative_path(
                configured_checkpoint, scenario_dir
            ))

    roles = resolve_agent_roles(scenario["agents"])
    if args.pretrained_actor and (args.eval or not roles.policy_agents or any(
            str(scenario["agents"][aid]["algorithm"]).lower() != "mappo" for aid in roles.policy_agents)):
        raise ValueError("--pretrained-actor requires MAPPO training; use --checkpoint for evaluation.")
    if scenario["experiment"].get("evaluation_only") and not args.eval:
        raise ValueError("This scenario is evaluation-only; pass --eval and --checkpoint.")
    if args.checkpoint and not args.eval and not roles.policy_agents:
        raise ValueError("--checkpoint for training currently supports one PPO learner only.")
    if args.eval:
        from application.evaluate import run_evaluation
        run_evaluation(scenario, args, console, scenario_dir)
    elif roles.policy_agents:
        from application.train import run_training
        run_training(scenario, args, console, scenario_dir, roles)
    else:
        from application.heuristic import run_heuristic
        run_heuristic(scenario, args, console)
