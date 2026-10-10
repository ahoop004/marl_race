from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from core.agent_roles import resolve_agent_roles
from core.environment_config import resolve_environment_config
from core.scenario import load_and_expand_scenario
from core.setup import build_obs_composers, build_reward_composers, create_environment_setup
from tasks import RaceTask
from wrappers.actions.composer import ActionComposer


def create_race_task(scenario, *, scenario_dir=None, mode="train", render_mode=None,
                     roles=None) -> RaceTask:
    if isinstance(scenario, (str, Path)):
        path = Path(scenario).resolve()
        scenario = load_and_expand_scenario(str(path))
        if scenario_dir is None:
            scenario_dir = path.parent
    scenario = deepcopy(scenario)
    scenario_dir = Path(scenario_dir) if scenario_dir is not None else Path.cwd()
    if render_mode not in (None, "human", "rgb_array"):
        raise ValueError(f"Unsupported render mode: {render_mode!r}")
    scenario["environment"]["render_mode"] = render_mode
    roles = roles or resolve_agent_roles(scenario["agents"])
    ids = list(roles.policy_agents)
    if not ids:
        raise ValueError("A racing task requires at least one policy agent")
    config = resolve_environment_config(scenario, mode=mode, scenario_dir=scenario_dir, roles=roles)
    env, controllers = create_environment_setup(scenario, mode=mode, scenario_dir=scenario_dir,
                                               roles=roles, env_config=config)
    try:
        repeat = config.get("action_repeat", 1)
        obs = build_obs_composers(scenario["agents"], ids, config, scenario_dir)
        rewards = build_reward_composers(scenario["agents"], ids, scenario_dir)
        actions = {
            aid: ActionComposer.from_config(
                env.action_spaces[aid].low, env.action_spaces[aid].high,
                scenario["agents"][aid].get("action_constraints", {}),
                decision_dt=env.timestep * repeat,
            ) for aid in ids
        }
        return RaceTask(
            env, policy_agents=ids, fixed_controllers=controllers, obs_composers=obs,
            reward_composers=rewards, action_composers=actions, action_repeat=repeat,
        )
    except BaseException:
        env.close()
        raise
