"""Inherited scenario loading and algorithm-independent task validation."""

from typing import Dict, Any, Optional
import copy
import math
import yaml
from pathlib import Path



class ScenarioError(Exception):
    """Exception raised for scenario configuration errors."""
    pass


def resolve_max_speed(scenario: Dict[str, Any]) -> Dict[str, Any]:
    """Apply an optional shared forward speed limit to learners and MPCs.

    The shared limit takes precedence over individual forward limits. With nonlinear
    physics it limits rolling-speed references, not the slipping chassis speed.
    Reverse bounds, acceleration limits and observation scales stay unchanged.
    """
    speed = scenario.get("environment", {}).get("max_speed")
    if speed is None:
        return scenario
    if (isinstance(speed, bool) or not isinstance(speed, (int, float))
            or not math.isfinite(speed) or speed <= 0):
        raise ScenarioError("environment.max_speed must be a positive finite number in m/s")

    result = copy.deepcopy(scenario)
    environment = result["environment"]
    vehicle = environment.setdefault("vehicle_params", {})
    if vehicle.get("model") == "combined_slip_st":
        actuators = vehicle.get("wheel_actuators", {})
        radius = actuators.get("wheel_radius")
        if (isinstance(radius, bool) or not isinstance(radius, (int, float))
                or not math.isfinite(radius) or radius <= 0):
            raise ScenarioError("environment.max_speed requires a positive finite wheel_radius")
        actuators["wheel_speed_max"] = float(speed) / radius
    else:
        vehicle["v_max"] = float(speed)
    from core.agent_builder import fixed_controller_names
    for agent in result.get("agents", {}).values():
        if str(agent.get("algorithm", "")).strip().lower() in fixed_controller_names():
            agent.setdefault("params", {})["max_speed"] = float(speed)
    return result


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-merge two dictionaries (override wins)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _load_yaml_file(path_obj: Path) -> Dict[str, Any]:
    """Load a YAML file and ensure it returns a dict."""
    if not path_obj.exists():
        raise FileNotFoundError(f"Config file not found: {path_obj}")

    try:
        with open(path_obj, 'r') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config {path_obj}: {e}") from e

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML dictionary: {path_obj}")
    return data


def load_yaml_config(path_obj: Path, visited: Optional[set] = None) -> Dict[str, Any]:
    """Load YAML includes relative to each file; later values override earlier ones."""
    path_obj = Path(path_obj).resolve()
    visited = visited or set()
    if path_obj in visited:
        raise ValueError(f"Include cycle detected at: {path_obj}")
    visited.add(path_obj)

    data = _load_yaml_file(path_obj)
    includes = data.pop('includes', None)

    merged: Dict[str, Any] = {}
    if includes:
        if isinstance(includes, (str, Path)):
            includes = [includes]
        if not isinstance(includes, list):
            raise ValueError("'includes' must be a list of file paths")
        for include_path in includes:
            if not isinstance(include_path, (str, Path)):
                raise ValueError("'includes' entries must be file paths")
            include_obj = (path_obj.parent / include_path).resolve()
            merged = _deep_merge(merged, load_yaml_config(include_obj, visited))

    merged = _deep_merge(merged, data)
    visited.remove(path_obj)
    return merged


def apply_parameter_overrides(scenario: Dict[str, Any], overrides) -> Dict[str, Any]:
    """Apply explicit dotted KEY=YAML choices; !delete removes an optional key.

    A YAML list can group assignment strings for sweeps. Mapping values replace
    the selected subtree. No scenario files are loaded.
    The caller validates the resulting configuration after other CLI overrides.
    """
    result = copy.deepcopy(scenario)
    assignments = []
    for item in overrides or ():
        if not isinstance(item, str):
            raise ScenarioError("Parameter overrides must be KEY=YAML strings")
        if item.lstrip().startswith("["):
            try:
                group = yaml.safe_load(item)
            except yaml.YAMLError as exc:
                raise ScenarioError("Invalid YAML override list") from exc
            if not isinstance(group, list) or any(not isinstance(value, str) for value in group):
                raise ScenarioError("An override list must contain KEY=YAML strings")
            assignments.extend(group)
        else:
            assignments.append(item)
    for item in assignments:
        key, separator, raw = item.partition("=")
        parts = key.split(".")
        if not separator or any(not part or not part.replace("_", "").isalnum() for part in parts):
            raise ScenarioError(f"Expected --set KEY=YAML with a dotted parameter name: {item!r}")
        target = result
        for part in parts[:-1]:
            if part not in target:
                target[part] = {}
            if not isinstance(target[part], dict):
                raise ScenarioError(f"Cannot set {key!r}: {part!r} is not a mapping")
            target = target[part]
        if raw.strip() == "!delete":
            target.pop(parts[-1], None)
        else:
            try:
                target[parts[-1]] = yaml.safe_load(raw)
            except yaml.YAMLError as exc:
                raise ScenarioError(f"Invalid YAML value for {key!r}: {raw!r}") from exc
    return result


def load_scenario(path: str) -> Dict[str, Any]:
    """Load scenario from YAML file.

    Args:
        path: Path to YAML scenario file

    Returns:
        Scenario configuration dict

    Raises:
        ScenarioError: If file not found or invalid YAML

    Example:
        >>> scenario = load_scenario('scenarios/ppo_lap_completion_pretrain.yaml')
        >>> scenario['experiment']['name']
        'ppo_lap_completion_pretrain'
    """
    path_obj = Path(path)

    try:
        return load_yaml_config(path_obj)
    except (OSError, ValueError) as exc:
        raise ScenarioError(str(exc)) from exc


def validate_scenario(scenario: Dict[str, Any]) -> None:
    """Validate physical/task configuration without selecting a learner."""
    from core.agent_builder import fixed_controller_names
    from core.agent_roles import resolve_agent_roles

    # --- Required top-level sections ---
    for section in ("experiment", "environment", "agents"):
        if section not in scenario:
            raise ScenarioError(f"Scenario must have a '{section}' section.")
        if not isinstance(scenario[section], dict):
            raise ScenarioError(f"Scenario '{section}' must be a mapping.")

    if "name" not in scenario["experiment"]:
        raise ScenarioError("'experiment' section must have a 'name' field.")
    environment = scenario["environment"]
    
    if any(key in block for block in (scenario, environment)
           for key in ("wheel_actuators", "combined_slip_vehicle")):
        raise ScenarioError(
            "wheel_actuators/combined_slip_vehicle is a component development profile, not a training "
            "configuration; configure environment.vehicle_params explicitly."
        )
    from physics.dynamic_models import validate_vehicle_params
    vehicle_params = environment.get("vehicle_params", environment.get("params", {}))
    if vehicle_params is not None:
        try:
            validate_vehicle_params(vehicle_params)
        except ValueError as exc:
            raise ScenarioError(str(exc)) from exc
    _MAP_KEYS = {"map", "maps", "map_bundle", "map_bundles"}
    if not _MAP_KEYS.intersection(environment):
        raise ScenarioError(
            "Environment must declare a map via one of: "
            + ", ".join(f"'{k}'" for k in sorted(_MAP_KEYS))
        )
    if "target_laps" in environment:
        target_laps = environment["target_laps"]
        if isinstance(target_laps, bool) or not isinstance(target_laps, int) or target_laps <= 0:
            raise ScenarioError("'environment.target_laps' must be a positive integer.")

    agents = scenario["agents"]
    from env.friction import validate_friction_protocol
    try:
        validate_friction_protocol(environment.get("friction"),
                                   nonlinear=(vehicle_params or {}).get("model") == "combined_slip_st")
    except ValueError as exc:
        raise ScenarioError(str(exc)) from exc
    if not isinstance(agents, dict) or not agents:
        raise ScenarioError("'agents' must be a non-empty dictionary.")

    teams = environment.get("agent_teams")
    if teams is not None and (not isinstance(teams, dict) or set(teams) != set(agents)
                              or any(not isinstance(team, str) or not team.strip() for team in teams.values())):
        raise ScenarioError("environment.agent_teams must assign every physical agent a nonempty team name.")

    try:
        roles = resolve_agent_roles(agents)
    except (ValueError, AttributeError) as exc:
        raise ScenarioError(str(exc)) from exc
    policy_ids = set(roles.policy_agents)
    for agent_id, agent_cfg in agents.items():
        if not isinstance(agent_cfg, dict):
            raise ScenarioError(f"Agent {agent_id!r} config must be a dictionary")
        if not str(agent_cfg.get("algorithm", "")).strip():
            raise ScenarioError(f"Agent {agent_id!r} requires an algorithm/controller name")
        if agent_id in policy_ids:
            for key in ("observation", "reward"):
                if key not in agent_cfg:
                    raise ScenarioError(f"Policy agent {agent_id!r} requires {key!r} config")
        else:
            if str(agent_cfg["algorithm"]).strip().lower() not in fixed_controller_names():
                raise ScenarioError(f"Unknown fixed controller {agent_cfg['algorithm']!r}")
        nonlinear = (vehicle_params or {}).get("model") == "combined_slip_st"
        action_mode = agent_cfg.get("action_constraints", {}).get("speed_control", "direct")
        if nonlinear and agent_id not in policy_ids:
            if agent_cfg.get("action_adapter") != "rolling_speed_to_wheel_v1":
                raise ScenarioError("Nonlinear fixed controllers require action_adapter: rolling_speed_to_wheel_v1")
            if action_mode != "direct":
                raise ScenarioError("Fixed wheel adapters require physical controller commands, not policy action constraints")
        elif agent_cfg.get("action_adapter") is not None:
            raise ScenarioError("action_adapter is supported only for nonlinear fixed controllers")
        if agent_id in policy_ids and nonlinear != (action_mode in {"wheel_speed", "wheel_acceleration"}):
            raise ScenarioError("combined_slip_st requires wheel_speed or wheel_acceleration actions; legacy uses vehicle-speed actions")
        if nonlinear and agent_id in policy_ids:
            from wrappers.actions.composer import ActionComposer
            constraints = agent_cfg.get("action_constraints", {})
            if constraints.get("speed_index", 1) != 1:
                raise ScenarioError("Wheel command must use speed_index 1")
            try:
                ActionComposer.contract_from_config(constraints,
                    float(environment.get("timestep", .01)) * int(environment.get("action_repeat", 1)))
            except ValueError as exc:
                raise ScenarioError(str(exc)) from exc

    lap_counting = environment.get("lap_counting", {}) or {}
    if not isinstance(lap_counting, dict):
        raise ScenarioError("environment.lap_counting must be a mapping")
    if "require_finish_line" in lap_counting and not isinstance(lap_counting["require_finish_line"], bool):
        raise ScenarioError("lap_counting.require_finish_line must be boolean")
    limits = environment.get("track_limits", {}) or {}
    if not isinstance(limits, dict) or set(limits) - {"enabled", "terminate"}:
        raise ScenarioError("track_limits accepts enabled and terminate booleans")
    if any(not isinstance(v, bool) for v in limits.values()):
        raise ScenarioError("track_limits values must be booleans")
    # Multi-car races can request boundary facts for rewards without enabling
    # the single-car time-trial boundary-reset protocol.
    if (limits.get("enabled") and limits.get("terminate", True)
            and (len(agents) != 1 or environment.get("terminate_on_collision", True))):
        raise ScenarioError("Track-limit time trials require one vehicle and terminate_on_collision: false")
    evaluation_mode = scenario.get("evaluation", {}).get("episode_termination_mode")
    if evaluation_mode is not None and evaluation_mode not in {"any_agent", "all_agents", "all_trainable"}:
        raise ScenarioError("evaluation.episode_termination_mode must be any_agent, all_agents, or all_trainable")


def load_and_expand_scenario(path: str, validate: bool = True, *, overrides=None) -> Dict[str, Any]:
    """Load inherited YAML, apply overrides and validate the scenario.

    The historical entry-point name is retained for callers.

    Args:
        path: Path to scenario YAML file
        validate: Whether to validate the scenario (default: True)
        overrides: Optional dotted KEY=YAML parameter choices, applied before validation.

    Returns:
        Expanded scenario with validated physical/task configuration

    Raises:
        ScenarioError: If scenario is invalid

    Example:
        >>> scenario = load_and_expand_scenario('scenarios/ppo_lap_completion_pretrain.yaml')
        >>> # Training entry points additionally validate learner eligibility
    """
    # Load raw scenario
    scenario = resolve_max_speed(apply_parameter_overrides(load_scenario(path), overrides))

    if validate:
        validate_scenario(scenario)

    return scenario


__all__ = [
    'ScenarioError',
    'load_scenario',
    'apply_parameter_overrides',
    'resolve_max_speed',
    'load_yaml_config',
    'validate_scenario',
    'load_and_expand_scenario',
]
