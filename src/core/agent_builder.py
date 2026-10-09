"""Build trainable-agent lists and fixed controllers from scenario config."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping


PYTORCH_RL_ALGOS = frozenset({"ppo", "mappo"})


def _create_racing_mpc(params: Dict[str, Any]) -> Any:
    from agents.mpc.racing import RacingMPCAgent
    return RacingMPCAgent(params)


_FIXED_CONTROLLERS: Dict[str, Callable] = {"racing_mpc": _create_racing_mpc}


def register_fixed_controller(name: str, constructor: Callable) -> None:
    name = str(name).strip().lower()
    if not name or name in PYTORCH_RL_ALGOS or not callable(constructor):
        raise ValueError("A fixed controller requires a non-RL name and callable constructor")
    _FIXED_CONTROLLERS[name] = constructor


def fixed_controller_names() -> tuple[str, ...]:
    return tuple(sorted(_FIXED_CONTROLLERS))


def create_fixed_controller(name: str, params: Dict[str, Any]) -> Any:
    key = str(name).strip().lower()
    if key not in _FIXED_CONTROLLERS:
        raise ValueError(f"Unknown fixed controller {name!r}; available: {fixed_controller_names()}")
    return _FIXED_CONTROLLERS[key](params)


def is_trainable_agent(agent_cfg: Mapping[str, Any]) -> bool:
    algo = str(agent_cfg.get("algorithm", "")).strip().lower()
    if algo not in PYTORCH_RL_ALGOS and algo not in _FIXED_CONTROLLERS:
        raise ValueError(f"Unsupported algorithm {algo!r}; expected PPO, MAPPO or a fixed controller")
    trainable = algo in PYTORCH_RL_ALGOS
    if "trainable" in agent_cfg and (
            not isinstance(agent_cfg["trainable"], bool) or agent_cfg["trainable"] != trainable):
        raise ValueError(f"Algorithm {algo!r} requires trainable={trainable}")
    return trainable


def get_trainable_agent_ids(agent_configs: Mapping[str, Mapping[str, Any]]) -> List[str]:
    return [aid for aid, cfg in agent_configs.items() if is_trainable_agent(cfg)]


def get_fixed_agent_ids(agent_configs: Mapping[str, Mapping[str, Any]]) -> List[str]:
    return [aid for aid, cfg in agent_configs.items() if not is_trainable_agent(cfg)]


def build_fixed_policy_agents(agent_configs: Mapping[str, Mapping[str, Any]], *, vehicle_params=None) -> Dict[str, Any]:
    agents = {}
    nonlinear = (vehicle_params or {}).get("model") == "combined_slip_st"
    for agent_id in get_fixed_agent_ids(agent_configs):
        config = agent_configs[agent_id]
        params = {**config.get("params", {}), "agent_id": agent_id}
        adapter = config.get("action_adapter")
        if nonlinear and adapter != "rolling_speed_to_wheel_v1":
            raise ValueError("Nonlinear fixed controllers require action_adapter: rolling_speed_to_wheel_v1")
        if not nonlinear and adapter is not None:
            raise ValueError("Wheel-command adapter requires combined_slip_st physics")
        controller = create_fixed_controller(config["algorithm"], params)
        if nonlinear:
            from wrappers.actions.composer import WheelReferenceAdapter
            controller = WheelReferenceAdapter(controller, vehicle_params["wheel_actuators"])
        agents[agent_id] = controller
    return agents
