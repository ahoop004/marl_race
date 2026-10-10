"""Action ownership resolved at the scenario boundary."""
from dataclasses import dataclass
from collections.abc import Mapping


# Older scenarios encode ownership solely in the algorithm field.
_LEGACY_POLICY_ALGORITHMS = frozenset({"ppo", "mappo"})


@dataclass(frozen=True)
class AgentRoles:
    policy_agents: tuple[str, ...]
    fixed_agents: tuple[str, ...]


def is_policy_agent(config: Mapping) -> bool:
    if not isinstance(config, Mapping):
        raise ValueError("Agent config must be a mapping")
    if "trainable" in config:
        if not isinstance(config["trainable"], bool):
            raise ValueError("Agent trainable must be boolean")
        return config["trainable"]
    algorithm = str(config.get("algorithm", "")).strip().lower()
    if algorithm in _LEGACY_POLICY_ALGORITHMS:
        return True
    from core.agent_builder import fixed_controller_names
    if algorithm in fixed_controller_names():
        return False
    raise ValueError(f"Agent {algorithm!r} requires explicit trainable action ownership")


def resolve_agent_roles(agent_configs: Mapping) -> AgentRoles:
    policy, fixed = [], []
    for aid, config in agent_configs.items():
        (policy if is_policy_agent(config) else fixed).append(aid)
    return AgentRoles(tuple(policy), tuple(fixed))
