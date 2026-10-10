"""Serial PettingZoo race collection and reporting for TorchRL MAPPO."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from env.types import TransitionRecord
from metrics.outcomes import determine_outcome
from training.hooks import TrainingHook, transition_record_hooks
from training.transition_records import transition_lifecycle_fields
from training.contracts import validate_team_reward_composers
from metrics.racing_eval import (team_finish_result, create_episode_facts,
                                update_agent_step_facts, finalize_episode_facts,
                                episode_race_record)
from tasks import RaceTask
from adapters import RaceParallelEnv


class TorchRLMAPPOTrainer:
    """Collect and report task decisions; the learner owns returns and updates."""

    def __init__(self, task: RaceTask, agent: Any, *,
                 hooks: Optional[List[TrainingHook]] = None, render: bool = False,
                 focal_agent_id: Optional[str] = None, run_id: str = "run") -> None:
        self.task = task
        self.agent = agent
        self.trainable_ids = list(task.possible_agents)
        self.other_agents = task.fixed_controllers
        self.action_repeat = task.action_repeat
        self.hooks = hooks or []
        self._environment_steps = self._physics_steps = self._agent_steps = self._updates = 0
        self._transition_hooks = transition_record_hooks(self.hooks)
        self.render = render
        self.focal_id = focal_agent_id or self.trainable_ids[0]
        if self.focal_id not in self.trainable_ids:
            raise ValueError("Focal agent must be a policy agent")
        self.run_id = run_id
        self.reward_mode = agent.reward_mode
        self.team_reward_reduction = agent.team_reward_reduction
        self.team_return_mode = agent.team_return_mode
        self._has_team_rewards = validate_team_reward_composers(
            task.reward_composers, trainable_ids=self.trainable_ids,
            opponent_ids=list(task.fixed_controllers), reward_mode=self.reward_mode,
            critic_mode=agent.critic_mode, team_return_mode=self.team_return_mode,
            action_repeat=self.action_repeat,
        )
        if self.team_return_mode == "joint" and self.action_repeat != 1:
            raise ValueError("Joint team returns currently require action_repeat=1")
        if list(agent.agent_ids) != self.trainable_ids:
            raise ValueError("Task policy agents must exactly match the MAPPO agent ID order")
        self.parallel_env = RaceParallelEnv(
            task, reward_mode=self.reward_mode,
            team_reward_reduction=self.team_reward_reduction,
        )

    def _episode_id(self, episode: int) -> str:
        return f"{self.run_id}_ep{episode:06d}"

    def _map_id(self):
        return self.task.episode_metadata.map_id

    def _spawn_id(self, agent_id):
        return self.task.episode_metadata.spawn_id(agent_id)

    def _reset_task(self):
        self.parallel_env.reset()
        return self.parallel_env.snapshot

    def _step_task(self, actions, on_physics_step):
        self.parallel_env.on_physics_step = on_physics_step
        if self.parallel_env.agents:
            self.parallel_env.step(actions)
        else:
            self.parallel_env.advance_fixed_agents()
        return self.parallel_env.last_step

    def _should_stop(self):
        return bool(getattr(self.agent, "should_stop", False)) or any(
            getattr(hook, "should_stop", False) for hook in self.hooks)

    def train(self, n_episodes: int = 0, *, total_steps: Optional[int] = None) -> None:
        """Train to an episode count or exact joint environment-decision budget."""
        if total_steps is not None and (isinstance(total_steps, bool)
                or not isinstance(total_steps, int) or total_steps <= 0):
            raise ValueError("total_steps must be a positive integer")
        collected, episode = 0, 0
        started_training = time.perf_counter()
        collection_started = started_training
        while (collected < total_steps if total_steps is not None else episode < n_episodes):
            if self._should_stop():
                break
            snapshot = self._reset_task()
            info_dict = snapshot.infos
            global_state = snapshot.global_state.vector
            self.agent.clear_buffers()
            wrapped_obs = dict(snapshot.observations)

            episode_done = False
            episode_reward = 0.0
            episode_truncated = False
            update_metrics: Dict = {}
            last_info: Dict = {}

            # Per-agent tracking — every trainable agent gets its own episode
            # reward total, last info payload, and truncation flag so each
            # agent's outcome can be reported independently of focal_id.
            agent_episode_rewards: Dict[str, float] = {aid: 0.0 for aid in self.trainable_ids}
            agent_individual_rewards: Dict[str, float] = {
                aid: 0.0 for aid in self.trainable_ids
            }
            agent_last_info: Dict[str, Dict] = {aid: {} for aid in self.trainable_ids}
            agent_truncated: Dict[str, bool] = {aid: False for aid in self.trainable_ids}
            step_idx = 0
            episode_id = self._episode_id(episode)
            map_id = self._map_id()
            for hook in self.hooks:
                callback = getattr(hook, 'on_episode_start', None)
                if callback is not None:
                    callback(dict(episode_id=episode_id, map_id=map_id,
                                  physics=info_dict.get(self.focal_id, {}).get('physics')))
            facts = create_episode_facts(episode=episode,
                agent_ids=[*self.trainable_ids, *self.other_agents],
                trainable_ids=self.trainable_ids, opponent_ids=list(self.other_agents))
            spawn_context = self.task.episode_metadata.spawn_configuration
            physics_steps = 0
            start_policy = getattr(self.agent, "policy_version", self._updates)
            finite_race = self.task.episode_limits.finish_on_laps
            team_components = {}

            def on_physics_step(substep):
                nonlocal physics_steps
                physics_steps += 1
                self._physics_steps += 1
                update_agent_step_facts(
                    facts, step_idx=physics_steps, infos=substep.facts.info,
                    terminations=substep.facts.terminations,
                    truncations=substep.facts.truncations, collect_speed=False,
                )
                if self.render:
                    try:
                        self.task.render()
                    except Exception:
                        pass

            while not episode_done and not self._should_stop():
                decision_policy = getattr(self.agent, "policy_version", self._updates)
                # --- Act: all trainable agents via shared actor ---
                active_trainable_ids = list(self.task.agents)
                rows = [wrapped_obs[aid] for aid in active_trainable_ids]
                stacked_observations = (self.agent.pack_observations(active_trainable_ids, rows)
                    if hasattr(self.agent, "pack_observations") else np.stack(rows)
                    if rows else np.empty((0, getattr(self.agent, "obs_dim", 0)), dtype=np.float32))
                if active_trainable_ids:
                    output = self.agent.sample_batch(active_trainable_ids, stacked_observations)
                    actions_norm, log_probs, raw_actions = output.actions, output.log_probs, output.raw_actions
                else:
                    actions_norm, log_probs, raw_actions = {}, {}, {}
                values = self.agent.evaluate_states(global_state, active_trainable_ids)
                task_step = self._step_task(actions_norm, on_physics_step=on_physics_step)
                info_dict = task_step.after.infos
                actions_phys = {aid: decision.action_physical for aid, decision in task_step.decisions.items()}
                accumulated_individual_rewards = {
                    aid: decision.individual_reward for aid, decision in task_step.decisions.items()
                }
                accumulated_learning_rewards = self.parallel_env.last_rewards
                reward_breakdowns = {aid: decision.reward_components for aid, decision in task_step.decisions.items()}
                decision_terminated = {aid: decision.terminated for aid, decision in task_step.decisions.items()}
                decision_truncated = {aid: decision.truncated for aid, decision in task_step.decisions.items()}
                team_breakdowns = task_step.team_reward_components
                for name, value in team_breakdowns.items():
                    team_components[name] = team_components.get(name, 0.0) + value
                team_step_reward = (next(iter(accumulated_learning_rewards.values()))
                    if self.team_return_mode == "joint" and accumulated_learning_rewards else 0.0)
                episode_done = task_step.after.episode_done
                episode_truncated = episode_truncated or decision_truncated.get(self.focal_id, False)
                post_step_global_snapshot = task_step.after.global_state
                next_global_state = post_step_global_snapshot.vector

                # --- Store transitions with accumulated rewards ---
                step_reward = 0.0
                next_wrapped_obs: Dict[str, np.ndarray] = {}
                agent_infos: Dict[str, Dict[str, Any]] = {}
                for aid in actions_norm:
                    reward = accumulated_learning_rewards.get(aid, 0.0)
                    individual_reward = accumulated_individual_rewards.get(aid, 0.0)
                    if aid == self.focal_id:
                        step_reward = reward

                    agent_info = dict(task_step.decisions[aid].info)
                    agent_info.update(
                        {
                            "individual_reward": individual_reward,
                            "learning_reward": reward,
                            "reward_mode": self.reward_mode,
                            "team_reward_reduction": self.team_reward_reduction,
                            "team_return_mode": self.team_return_mode,
                        }
                    )
                    next_obs = task_step.decisions[aid].next_observation
                    next_wrapped_obs[aid] = next_obs
                    agent_infos[aid] = agent_info

                    agent_episode_rewards[aid] += reward
                    agent_individual_rewards[aid] += individual_reward
                    facts.agents[aid].reward_total += reward
                    facts.agents[aid].individual_reward_total += individual_reward
                    for name, value in reward_breakdowns[aid].items():
                        components = facts.agents[aid].reward_components
                        components[name] = components.get(name, 0.0) + value
                    agent_last_info[aid] = agent_info
                    agent_truncated[aid] = (
                        agent_truncated[aid] or decision_truncated[aid]
                    )

                ordered_ids = list(actions_norm)
                self.agent.store_batch(
                    ordered_ids,
                    observations=wrapped_obs,
                    global_state=global_state,
                    actions=actions_norm,
                    rewards=accumulated_learning_rewards,
                    log_probs=log_probs,
                    values=values,
                    terminated=decision_terminated,
                    truncated=decision_truncated,
                    raw_actions=raw_actions,
                )
                if self.team_return_mode == "joint" and ordered_ids:
                    self.agent.store_team_step(
                        ordered_ids, reward=team_step_reward,
                        value=values[ordered_ids[0]],
                        terminal=not self.task.agents,
                    )
                if ordered_ids and hasattr(self.agent, "set_next_state"):
                    self.agent.set_next_state(next_global_state)

                if self._transition_hooks:
                    for aid in ordered_ids:
                        record = TransitionRecord(
                            obs=wrapped_obs[aid],
                            action_norm=actions_norm[aid],
                            action_phys=actions_phys[aid],
                            reward=accumulated_learning_rewards[aid],
                            reward_components={**reward_breakdowns[aid], **team_breakdowns},
                            next_obs=next_wrapped_obs[aid],
                            terminated=decision_terminated[aid],
                            truncated=decision_truncated[aid],
                            info=agent_infos[aid],
                            global_state=np.asarray(
                                global_state, dtype=np.float32
                            ).copy(),
                            map_id=map_id,
                            spawn_id=self._spawn_id(aid),
                            episode_id=episode_id,
                            step_idx=step_idx,
                            agent_id=aid,
                            **transition_lifecycle_fields(
                                agent_infos[aid],
                                global_state=post_step_global_snapshot,
                            ),
                        )
                        for hook in self._transition_hooks:
                            hook.on_step(record)

                episode_reward += team_step_reward if self.team_return_mode == "joint" else step_reward

                wrapped_obs = dict(task_step.after.observations)

                global_state = next_global_state
                step_idx += 1
                self._environment_steps += 1
                self._agent_steps += len(ordered_ids)
                collected += 1
                budget_done = total_steps is not None and collected >= total_steps

                # --- Trigger update when any buffer is full or episode ends ---
                if self.agent.any_buffer_full() or episode_done or budget_done:
                    rollout_samples = sum(buf.size() for buf in self.agent.buffers.values())
                    collection_seconds = time.perf_counter() - collection_started
                    update_started = time.perf_counter()
                    update_metrics = self.agent.update(
                        next_global_state=next_global_state,
                    )
                    self._updates += bool(update_metrics)
                    update_metrics["perf/update_seconds"] = time.perf_counter() - update_started
                    update_metrics["perf/collection_seconds"] = collection_seconds
                    update_metrics["perf/elapsed_seconds"] = time.perf_counter() - started_training
                    update_metrics["train/rollout_agent_samples"] = rollout_samples
                    update_metrics["perf/end_to_end_env_steps_per_second"] = collected / max(
                        time.perf_counter() - started_training, 1e-9)
                    self.agent.clear_buffers()
                    update_metrics["train/environment_steps"] = self._environment_steps
                    update_metrics["train/physics_steps"] = self._physics_steps
                    update_metrics["train/agent_steps"] = self._agent_steps
                    update_metrics["train/updates"] = self._updates
                    for hook in self.hooks:
                        hook.on_update(update_metrics)
                    collection_started = time.perf_counter()

                if budget_done:
                    break

            # A budget cut is neither a crash nor a completed episode. The last
            # fragment was bootstrapped above; do not fabricate episode metrics.
            if not episode_done:
                break
            last_info = agent_last_info.get(self.focal_id, {})
            outcome = determine_outcome(last_info, truncated=episode_truncated)
            last_info["outcome"] = outcome.value

            # Per-agent outcome/reward breakdown — every trainable agent's
            # info carries its own target_id-relative collision/finish flags,
            # so each agent's outcome is determined independently.
            agent_outcomes = {
                aid: determine_outcome(
                    agent_last_info.get(aid, {}), truncated=agent_truncated.get(aid, False)
                ).value
                for aid in self.trainable_ids
            }
            episode_metrics = dict(update_metrics)
            episode_metrics["episode_steps"] = step_idx
            episode_metrics["agent_rewards"] = dict(agent_episode_rewards)
            episode_metrics["agent_individual_rewards"] = dict(
                agent_individual_rewards
            )
            episode_metrics["reward_mode"] = self.reward_mode
            episode_metrics["team_reward_reduction"] = self.team_reward_reduction
            if self.team_return_mode == "joint":
                episode_metrics["team_episode_reward"] = episode_reward
            if self._has_team_rewards:
                episode_metrics.update({
                    f"team_result/{key}": value for key, value in team_finish_result(
                        info_dict, self.trainable_ids, list(self.other_agents)
                    ).items()
                })
            episode_metrics["agent_outcomes"] = agent_outcomes
            episode_metrics["agent_terminal_reasons"] = {
                aid: agent_last_info.get(aid, {}).get("terminal_reason")
                for aid in self.trainable_ids
            }
            episode_metrics["agent_finish_positions"] = {
                aid: agent_last_info.get(aid, {}).get("finish_position")
                for aid in self.trainable_ids
            }
            episode_metrics["agent_lap_counts"] = {
                aid: int(agent_last_info.get(aid, {}).get("lap_count", 0))
                for aid in self.trainable_ids
            }
            episode_metrics["race_record"] = {
                **episode_race_record(finalize_episode_facts(facts),
                    timestep=self.task.timestep, finite_race=finite_race),
                "run_id": self.run_id, "environment_id": 0, "episode_id": episode_id,
                "environment_episode": episode, "map_id": map_id,
                "environment_seed": self.task.episode_metadata.environment_seed,
                "spawn_ids": {aid: self._spawn_id(aid) for aid in facts.agents},
                "spawn_configuration": spawn_context,
                "environment_decisions": step_idx,
                "environment_decisions_total": self._environment_steps,
                "action_repeat": self.action_repeat,
                "policy_version_start": start_policy,
                "policy_version_end": decision_policy,
                "reported_at_environment_steps": self._environment_steps,
                "team_reward_components": team_components,
                "training_return": episode_reward,
            }

            for hook in self.hooks:
                hook.on_episode_end(episode, episode_reward, last_info, episode_metrics)
            episode += 1

        for hook in self.hooks:
            hook.on_training_end()
