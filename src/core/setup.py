"""Build physical environments, fixed controllers and task composers."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from env.RaceEnv import RaceEnv
from core.agent_builder import build_fixed_policy_agents
from core.env_builder import create_environment
from core.agent_roles import AgentRoles, resolve_agent_roles
from core.environment_config import resolve_environment_config
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
    env_config: Optional[Dict[str, Any]] = None,
) -> Tuple[RaceEnv, Dict[str, Any]]:
    """Create physics and fixed controllers from resolved action-owner roles.

    Resolve physical configuration when the caller has not already supplied it.
    This builder neither selects a learner nor seeds policy randomness.
    """
    env_config = (resolve_environment_config(
        scenario, mode=mode, scenario_dir=scenario_dir, roles=roles,
    ) if env_config is None else env_config)
    agent_configs = scenario['agents']
    roles = roles or resolve_agent_roles(agent_configs)
    seed = scenario['experiment'].get('seed')
    env = create_environment(env_config, agent_configs, seed)

    try:
        # Preserve explicitly configured targets in environment lifecycle facts.
        target_mapping = {
            aid: cfg["target_id"]
            for aid, cfg in agent_configs.items()
            if cfg.get("target_id")
        }
        if target_mapping:
            env.configure_agent_targets(target_mapping)

        agents = build_fixed_policy_agents(agent_configs, fixed_ids=roles.fixed_agents,
                                          vehicle_params=env.params)
        for controller in agents.values():
            if hasattr(controller, "set_env"):
                controller.set_env(env)
    except BaseException:
        env.close()
        raise
    return env, agents
