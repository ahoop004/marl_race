"""One task-driven race runner for checkpoint selection and standalone evaluation."""
from dataclasses import dataclass
from typing import Any

from adapters.rewards import RewardMapping
from metrics.racing_eval import (
    create_episode_facts, update_agent_step_facts, finalize_episode_facts,
    episode_race_record,
)


@dataclass(frozen=True)
class EvaluationEpisode:
    facts: Any
    snapshot: Any
    map_id: str
    spawn_context: dict
    record: dict
    physics: Any
    team_return: float
    team_components: dict


def run_evaluation_episode(task, actions_fn, *, episode, seed,
                           reward_mapping=RewardMapping(), completion="race",
                           render=False, progress=None, include_rewards=True):
    """Advance the physical race or stop when all exposed policy agents retire.

    actions_fn sees local composed observations only. Reward composers, physical
    actions, fixed controllers and resets remain exclusively owned by RaceTask.
    """
    if completion not in {"race", "policy"}:
        raise ValueError("Evaluation completion must be race or policy")
    snapshot = task.reset(seed=seed, options={"map_episode_index": episode,
                                           "spawn_episode_index": episode})
    spawn = task.episode_metadata.spawn_configuration
    map_id = task.episode_metadata.map_id
    facts = create_episode_facts(episode=episode, agent_ids=task.physical_agents,
                                trainable_ids=task.possible_agents,
                                opponent_ids=task.fixed_policy_agents)
    clean = dict.fromkeys(task.possible_agents, True)
    team_return, team_components = 0.0, {}

    def observe(substep):
        raw = substep.facts
        update_agent_step_facts(facts, step_idx=substep.physics_step, infos=raw.info,
                                terminations=raw.terminations, truncations=raw.truncations,
                                agent_states=raw.agent_states)
        for aid in clean:
            info = raw.info.get(aid, {})
            clean[aid] &= not (info.get("collision", False)
                               or (info.get("track_limits") or {}).get("exceeded", False))
        if render:
            task.render()

    if progress:
        progress(facts, episode, 0, "starting")
    while not snapshot.episode_done and (completion == "race" or snapshot.agents):
        actions = actions_fn(snapshot.agents, snapshot.observations) if snapshot.agents else {}
        step = task.step(actions, on_physics_step=observe)
        rewards = reward_mapping.from_step(step, task.possible_agents)
        if reward_mapping.mode == "team_shared" and rewards:
            team_return += next(iter(rewards.values()))
        for aid, decision in step.decisions.items():
            agent = facts.agents[aid]
            agent.individual_reward_total += decision.individual_reward
            agent.reward_total += rewards[aid]
            for name, value in decision.reward_components.items():
                agent.reward_components[name] = agent.reward_components.get(name, 0.0) + value
        for name, value in step.team_reward_components.items():
            team_components[name] = team_components.get(name, 0.0) + value
        snapshot = step.after
        if progress:
            progress(facts, episode, snapshot.physics_steps, "running")
    finalize_episode_facts(facts)
    if progress:
        progress(facts, episode, snapshot.physics_steps, "complete")
    record = {
        **episode_race_record(facts, timestep=task.timestep, include_rewards=include_rewards),
        "phase": "evaluation", "environment_episode": episode,
        "seed": seed, "map_id": map_id, "spawn_configuration": spawn,
        "strict_clean_finish": all(clean[aid] and facts.agents[aid].clean_finish
                                   for aid in task.possible_agents),
    }
    if include_rewards:
        record.update(reward_mode=reward_mapping.mode,
                      team_reward_reduction=reward_mapping.reduction,
                      team_reward_components=team_components)
        if reward_mapping.mode == "team_shared":
            record["team_episode_reward"] = team_return
    physics = snapshot.infos.get(task.possible_agents[0], {}).get("physics")
    return EvaluationEpisode(facts, snapshot, map_id, spawn, record, physics,
                             team_return, team_components)
