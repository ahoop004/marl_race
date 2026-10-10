"""Resolve physical configuration before constructing physics or task composers."""
from copy import deepcopy
from pathlib import Path

from core.agent_roles import resolve_agent_roles
from core.feature_requirements import derive_environment_feature_requirements
from core.map_selection import apply_map_split


def resolve_environment_config(scenario, *, mode="train", scenario_dir=None, roles=None):
    """Return a detached config shared by environment and composer construction."""
    experiment_config = scenario['experiment']
    env_config = deepcopy(scenario['environment'])
    env_config = apply_map_split(env_config, experiment_config, mode)
    env_config["physics_phase"] = "eval" if mode in {"eval", "evaluation", "test"} else "train"
    evaluation = scenario.get("evaluation", {}) or {}
    if env_config["physics_phase"] == "eval" and "no_progress" in evaluation:
        env_config["no_progress"] = evaluation["no_progress"]
    if env_config["physics_phase"] == "eval" and "target_laps" in evaluation:
        # Continuous training can disable finishing and timeouts. Evaluation
        # explicitly restores a finite race for PPO and multi-agent MAPPO alike.
        env_config["target_laps"] = int(evaluation["target_laps"])
        env_config["episode_termination"] = {**env_config.get("episode_termination", {}),
                                             "lap_completion": True}
        if "max_steps" in evaluation:
            env_config["max_steps"] = int(evaluation["max_steps"])
    if env_config["physics_phase"] == "eval" and "terminate_on_collision" in evaluation:
        env_config["terminate_on_collision"] = evaluation["terminate_on_collision"]
    if env_config["physics_phase"] == "eval" and env_config.get("track_limits", {}).get("enabled"):
        # Default paper evaluation records excursions; safety evaluation can
        # explicitly retain boundary termination as well as collision checks.
        env_config["track_limits"] = {
            "enabled": True,
            "terminate": bool(evaluation.get("terminate_on_track_limit", False)),
        }
        env_config["episode_termination"] = {**env_config.get("episode_termination", {}),
                                             "lap_completion": True}
        env_config["target_laps"] = int(evaluation.get("target_laps", 20))
        env_config["max_steps"] = int(evaluation.get("max_steps", 16000))
    if env_config["physics_phase"] == "eval" and "lap_completion" in evaluation:
        env_config["episode_termination"] = {**env_config.get("episode_termination", {}),
                                             "lap_completion": evaluation["lap_completion"]}
    if env_config["physics_phase"] == "eval" and "episode_termination_mode" in evaluation:
        env_config["episode_termination"] = {**env_config.get("episode_termination", {}),
                                             "mode": evaluation["episode_termination_mode"]}
    agent_configs = scenario['agents']
    roles = roles or resolve_agent_roles(agent_configs)
    if (set(roles.policy_agents) & set(roles.fixed_agents)
            or set(roles.policy_agents) | set(roles.fixed_agents) != set(agent_configs)):
        raise ValueError("Every physical agent needs exactly one action owner")
    env_config.setdefault("trainable_agents", list(roles.policy_agents))
    env_config.setdefault("fixed_policy_agents", list(roles.fixed_agents))
    if scenario_dir is not None:
        requirements = derive_environment_feature_requirements(
            agent_configs,
            scenario_dir=Path(scenario_dir),
            centerline_render=bool(env_config.get("centerline_render", False)),
        )
        env_config["feature_requirements"] = requirements.as_dict()
        geometry_required = bool(
            requirements.requires_centerline_facts
            or requirements.requires_track_preview
            or requirements.requires_frenet_neighbors
            or requirements.centerline_render
        )
        if geometry_required:
            env_config["centerline_autoload"] = True
        if (
            requirements.requires_centerline_facts
            or requirements.requires_track_preview
            or requirements.requires_frenet_neighbors
        ):
            env_config["centerline_features"] = True

    return env_config
