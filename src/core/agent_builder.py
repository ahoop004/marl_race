"""Build trainable-agent lists and fixed controllers from scenario config."""
from __future__ import annotations

from typing import Any, Callable, Dict

from core.agent_roles import resolve_agent_roles


def _create_racing_mpc(params: Dict[str, Any]) -> Any:
    from agents.mpc.racing import RacingMPCAgent
    return RacingMPCAgent(params)


_FIXED_CONTROLLERS: Dict[str, Callable] = {"racing_mpc": _create_racing_mpc}


def register_fixed_controller(name: str, constructor: Callable) -> None:
    name = str(name).strip().lower()
    if not name or not callable(constructor):
        raise ValueError("A fixed controller requires a non-RL name and callable constructor")
    _FIXED_CONTROLLERS[name] = constructor


def fixed_controller_names() -> tuple[str, ...]:
    return tuple(sorted(_FIXED_CONTROLLERS))


def create_fixed_controller(name: str, params: Dict[str, Any]) -> Any:
    key = str(name).strip().lower()
    if key not in _FIXED_CONTROLLERS:
        raise ValueError(f"Unknown fixed controller {name!r}; available: {fixed_controller_names()}")
    return _FIXED_CONTROLLERS[key](params)


def build_fixed_policy_agents(agent_configs: Mapping[str, Mapping[str, Any]], *,
                             fixed_ids=None, vehicle_params=None) -> Dict[str, Any]:
    agents = {}
    nonlinear = (vehicle_params or {}).get("model") == "combined_slip_st"
    for agent_id in (resolve_agent_roles(agent_configs).fixed_agents if fixed_ids is None else fixed_ids):
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
