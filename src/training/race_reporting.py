"""Racing lifecycle reports subscribed to native environment events.

No policy execution, trajectory storage, or optimization belongs here.
"""
from metrics.outcomes import determine_outcome
from metrics.racing_eval import (create_episode_facts, update_agent_step_facts,
                                finalize_episode_facts, episode_race_record,
                                team_finish_result)
from env.types import TransitionRecord
from training.hooks import transition_record_hooks
from training.transition_records import transition_lifecycle_fields


class RaceTrainingReports:
    def __init__(self, trainer):
        self.trainer = trainer
        self.task, self.agent = trainer.task, trainer.agent
        self.ids = list(self.task.possible_agents)
        self.transition_hooks = transition_record_hooks(trainer.hooks)
        self.completed = 0
        self.pending_episodes = []

    def reset(self, snapshot):
        t = self.trainer
        # Collector eagerly resets after terminal transitions, even on its last
        # frame. Do not report a new episode when the experiment has ended.
        if t.budget_reached() or t._should_stop():
            return
        self.episode = self.completed
        self.episode_id = f"{t.run_id}_ep{self.episode:06d}"
        metadata = self.task.episode_metadata
        self.map_id = metadata.map_id
        self.spawn_ids = {aid: metadata.spawn_id(aid) for aid in self.task.physical_agents}
        self.environment_seed = metadata.environment_seed
        self.spawn_context = metadata.spawn_configuration
        self.start_policy = t._updates
        self.steps = self.physics_steps = 0
        self.reward = 0.0
        self.rewards = dict.fromkeys(self.ids, 0.0)
        self.individual_rewards = dict(self.rewards)
        self.last_info = {aid: {} for aid in self.ids}
        self.truncated = dict.fromkeys(self.ids, False)
        self.team_components = {}
        self.facts = create_episode_facts(
            episode=self.episode, agent_ids=list(self.task.physical_agents),
            trainable_ids=self.ids, opponent_ids=list(self.task.fixed_controllers))
        for hook in t.hooks:
            callback = getattr(hook, "on_episode_start", None)
            if callback is not None:
                callback(dict(episode_id=self.episode_id, map_id=self.map_id,
                              physics=snapshot.infos.get(t.focal_id, {}).get("physics")))

    def physics(self, substep):
        self.physics_steps += 1
        self.trainer._physics_steps += 1
        update_agent_step_facts(self.facts, step_idx=self.physics_steps,
                               infos=substep.facts.info,
                               terminations=substep.facts.terminations,
                               truncations=substep.facts.truncations, collect_speed=False)

    def decision(self, result):
        t = self.trainer
        if self.steps == 0:
            self.start_policy = t._updates
        rewards = t.env.reward_mapping.from_step(result, self.ids)
        team_reward = next(iter(rewards.values()), 0.0)
        self.reward += team_reward if t.algorithm == "mappo" else sum(rewards.values())
        for name, value in result.team_reward_components.items():
            self.team_components[name] = self.team_components.get(name, 0.0) + value
        for aid, decision in result.decisions.items():
            reward = rewards[aid]
            info = dict(decision.info)
            if t.algorithm == "mappo":
                info.update(individual_reward=decision.individual_reward, learning_reward=reward,
                            reward_mode=self.agent.reward_mode,
                            team_reward_reduction=self.agent.team_reward_reduction,
                            team_return_mode=self.agent.team_return_mode)
            self.rewards[aid] += reward
            self.individual_rewards[aid] += decision.individual_reward
            self.last_info[aid] = info
            self.truncated[aid] |= decision.truncated
            fact = self.facts.agents[aid]
            fact.reward_total += reward
            fact.individual_reward_total += decision.individual_reward
            for name, value in decision.reward_components.items():
                fact.reward_components[name] = fact.reward_components.get(name, 0.0) + value
            if self.transition_hooks:
                record = TransitionRecord(
                    obs=decision.observation, action_norm=decision.action_normalized,
                    action_phys=decision.action_physical, reward=reward,
                    reward_components={**decision.reward_components, **result.team_reward_components},
                    next_obs=decision.next_observation, terminated=decision.terminated,
                    truncated=decision.truncated, info=info,
                    global_state=result.before.global_state.vector.copy(),
                    map_id=self.map_id, spawn_id=self.spawn_ids[aid],
                    episode_id=self.episode_id, step_idx=self.steps, agent_id=aid,
                    **transition_lifecycle_fields(info, global_state=result.after.global_state))
                for hook in self.transition_hooks:
                    hook.on_step(record)
        self.steps += 1
        t._environment_steps += 1
        t._agent_steps += result.agent_steps
        t._pending_learning_steps += bool(result.decisions)
        if result.after.episode_done:
            self.finish(result)

    def finish(self, result):
        t = self.trainer
        info = dict(self.last_info[t.focal_id])
        info["outcome"] = determine_outcome(info, truncated=self.truncated[t.focal_id]).value
        info.setdefault("map_bundle", self.map_id)
        metrics = dict(episode_steps=self.steps)
        lap_steps = info.get("lap_time_steps")
        metrics["lap_time_s"] = float(lap_steps) * self.task.timestep if lap_steps is not None else None
        metrics.update(
            agent_rewards=dict(self.rewards), agent_individual_rewards=dict(self.individual_rewards),
            agent_outcomes={aid: determine_outcome(self.last_info[aid], truncated=self.truncated[aid]).value
                            for aid in self.ids},
            agent_terminal_reasons={aid: self.last_info[aid].get("terminal_reason") for aid in self.ids},
            agent_finish_positions={aid: self.last_info[aid].get("finish_position") for aid in self.ids},
            agent_lap_counts={aid: int(self.last_info[aid].get("lap_count", 0)) for aid in self.ids})
        if t.algorithm == "mappo":
            metrics.update(reward_mode=self.agent.reward_mode,
                           team_reward_reduction=self.agent.team_reward_reduction,
                           team_episode_reward=self.reward)
            metrics.update({f"team_result/{key}": value for key, value in team_finish_result(
                result.after.infos, self.ids, list(self.task.fixed_controllers)).items()})
        metrics["race_record"] = {
            **episode_race_record(finalize_episode_facts(self.facts), timestep=self.task.timestep,
                                 finite_race=self.task.episode_limits.finish_on_laps),
            "run_id": t.run_id, "environment_id": 0, "episode_id": self.episode_id,
            "environment_episode": self.episode, "map_id": self.map_id,
            "environment_seed": self.environment_seed, "spawn_ids": self.spawn_ids,
            "spawn_configuration": self.spawn_context, "environment_decisions": self.steps,
            "environment_decisions_total": t._environment_steps,
            "action_repeat": self.task.action_repeat,
            "policy_version_start": self.start_policy, "policy_version_end": t._updates,
            "reported_at_environment_steps": t._environment_steps,
            "team_reward_components": dict(self.team_components), "training_return": self.reward}
        if t.algorithm == "ppo":
            # Preserve PPO's existing episode CSV path and metric contract.
            metrics.pop("race_record")
        self.pending_episodes.append((self.episode, self.reward, info, metrics))
        self.completed += 1

    def dispatch(self, update_metrics):
        for episode, reward, info, metrics in self.pending_episodes:
            metrics = {**update_metrics, **metrics}
            for hook in self.trainer.hooks:
                hook.on_episode_end(episode, reward, info, metrics)
        self.pending_episodes.clear()
