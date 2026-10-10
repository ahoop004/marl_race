from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from core.agent_builder import get_trainable_agent_ids
from core.scenario import load_and_expand_scenario, resolve_mappo_config
from core.setup import build_obs_composers, build_reward_composers, create_training_setup
from tasks import RaceTask
from tasks.reward_context import validate_team_reward_composers
from wrappers.actions.composer import ActionComposer


def create_race_task(scenario, *, scenario_dir=None, mode="train", render_mode=None) -> RaceTask:
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
    ids = get_trainable_agent_ids(scenario["agents"])
    if not ids:
        raise ValueError("A racing task requires at least one policy agent")
    env, controllers, _ = create_training_setup(scenario, mode=mode, scenario_dir=scenario_dir)
    try:
        config = scenario["environment"]
        repeat = config.get("action_repeat", 1)
        obs = build_obs_composers(scenario["agents"], ids, config, scenario_dir)
        rewards = build_reward_composers(scenario["agents"], ids, scenario_dir)
        params = {**scenario.get("training_defaults", {}), **scenario["agents"][ids[0]].get("params", {})}
        mappo = resolve_mappo_config(scenario)
        has_team_rewards = validate_team_reward_composers(
            rewards, trainable_ids=ids, opponent_ids=list(controllers),
            reward_mode=mappo["reward_mode"], critic_mode=mappo["critic_mode"],
            team_return_mode=params.get("team_return_mode", "per_agent"), action_repeat=repeat,
        )
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
            team_reward_agent_id=ids[0] if has_team_rewards else None,
        )
    except BaseException:
        env.close()
        raise
