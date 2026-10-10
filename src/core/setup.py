"""Build physical environments, fixed controllers and task composers."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from env.RaceEnv import RaceEnv
from core.agent_builder import (
    build_fixed_policy_agents,
    AgentRoles, resolve_agent_roles,
)
from core.env_builder import create_environment
from core.feature_requirements import derive_environment_feature_requirements
from core.map_selection import apply_map_split
from wrappers.observations.composer import ObservationComposer
from wrappers.rewards.composer import RewardComposer


def build_obs_composer(
    agent_cfg: Dict, env_config: Dict, scenario_dir: Path
) -> ObservationComposer:
    """Build one ObservationComposer for a single agent config."""
    obs_ref = agent_cfg.get("observation")
    if isinstance(obs_ref, str):
        obs_path = (scenario_dir / obs_ref).resolve()
        return ObservationComposer.from_file(str(obs_path), env_config)
    elif isinstance(obs_ref, dict):
        return ObservationComposer.from_config(obs_ref, env_config)
    raise ValueError("Agent 'observation' must be a file path or inline config dict.")


def build_reward_composer(agent_cfg: Dict, scenario_dir: Path) -> RewardComposer:
    """Build one RewardComposer for a single agent config."""
    reward_ref = agent_cfg.get("reward")
    if isinstance(reward_ref, str):
        reward_path = (scenario_dir / reward_ref).resolve()
        return RewardComposer.from_file(str(reward_path))
    elif isinstance(reward_ref, dict):
        return RewardComposer.from_config(reward_ref)
    raise ValueError("Agent 'reward' must be a file path or inline config dict.")


def build_obs_composers(
    agent_configs: Dict,
    trainable_ids: List[str],
    env_config: Dict,
    scenario_dir: Path,
) -> Dict[str, ObservationComposer]:
    """Build one ObservationComposer per trainable agent.

    Returns a dict keyed by agent_id.  Single-agent trainers index into it
    with their ``rl_agent_id``; MAPPO iterates over all entries.
    """
    return {
        aid: build_obs_composer(agent_configs[aid], env_config, scenario_dir)
        for aid in trainable_ids
    }


def build_reward_composers(
    agent_configs: Dict,
    trainable_ids: List[str],
    scenario_dir: Path,
) -> Dict[str, RewardComposer]:
    """Build one RewardComposer per trainable agent.

    Returns a dict keyed by agent_id.
    """
    return {
        aid: build_reward_composer(agent_configs[aid], scenario_dir)
        for aid in trainable_ids
    }


def create_environment_setup(
    scenario: Dict[str, Any],
    *,
    mode: str = "train",
    scenario_dir: Optional[Path] = None,
    roles: Optional[AgentRoles] = None,
) -> Tuple[RaceEnv, Dict[str, Any]]:
    """Create physics and fixed controllers from resolved action-owner roles.

    Mode selects map splits and physical evaluation overrides. This builder
    neither selects a learner nor seeds process-global policy randomness.
    """
    # Extract configuration sections
    experiment_config = scenario['experiment']
    env_config = dict(scenario['environment'])
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

    seed = experiment_config.get('seed')
    env = create_environment(env_config, agent_configs, seed)

    # Preserve explicitly configured targets in environment lifecycle facts.
    target_mapping = {
        aid: cfg["target_id"]
        for aid, cfg in agent_configs.items()
        if cfg.get("target_id")
    }
    if target_mapping:
        env.configure_agent_targets(target_mapping)

    try:
        agents = build_fixed_policy_agents(agent_configs, fixed_ids=roles.fixed_agents,
                                          vehicle_params=env.params)
        for controller in agents.values():
            if hasattr(controller, "set_env"):
                controller.set_env(env)
    except Exception:
        env.close()
        raise
    return env, agents
