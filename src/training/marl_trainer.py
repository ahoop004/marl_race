"""Multi-agent training loop for MAPPO (and future MARL algorithms).

``MARLTrainer`` drives an episode-based training loop for scenarios with
multiple trainable agents sharing one :class:`~agents.mappo.MAPPOAgent`.

Key differences from :class:`~training.on_policy_trainer.OnPolicyTrainer`
--------------------------------------------------------------------------
- Actions are collected for **all** trainable agents each step via the
  shared actor.  Fixed-policy opponents are still polled via ``other_agents``.
- The centralized critic uses ``env.get_global_state().vector`` — a flat
  concatenation of all agents' state — rather than local observations.
- Per-agent factual rewards are computed independently, then either retained
  or reduced to one shared team learning reward according to the MAPPO config.
- The buffer-full / update trigger is checked across all agents: when **any**
  agent's buffer is full, an update is triggered for all.
"""
from __future__ import annotations

import copy
import time
from typing import Any, Dict, List, Optional

import numpy as np

from agents.mappo import MAPPOAgent
from env.types import TransitionRecord
from metrics.outcomes import determine_outcome
from training.hooks import TrainingHook, transition_record_hooks
from training.reward_context import transition_lifecycle_fields, validate_team_reward_composers
from metrics.racing_eval import (team_finish_result, create_episode_facts,
                                update_agent_step_facts, finalize_episode_facts,
                                episode_race_record, capture_spawn_context)
from tasks import RaceTask
from wrappers.actions.composer import ActionComposer
from wrappers.observations.composer import ObservationComposer
from wrappers.rewards.composer import RewardComposer


def map_mappo_learning_rewards(
    individual_rewards: Dict[str, float],
    *,
    trainable_ids: List[str],
    reward_mode: str,
    team_reward_reduction: str,
) -> Dict[str, float]:
    """Convert per-agent factual rewards into MAPPO learning rewards."""
    if reward_mode == "individual":
        return dict(individual_rewards)
    team_reward = float(sum(individual_rewards.values()))
    if team_reward_reduction == "mean":
        team_reward /= max(len(trainable_ids), 1)
    return {aid: team_reward for aid in individual_rewards}


class MARLTrainer:
    """Episode-based MAPPO training loop.

    Parameters
    ----------
    env:
        The F110 parallel environment.
    agent:
        A :class:`~agents.mappo.MAPPOAgent` with shared actor and centralized critic.
    trainable_ids:
        Ordered list of agent IDs that are being trained.
    other_agents:
        Dict of ``{agent_id: heuristic_policy}`` for fixed-policy opponents.
    obs_composers:
        ``{agent_id: ObservationComposer}`` — one per trainable agent.
    reward_composers:
        ``{agent_id: RewardComposer}`` — one per trainable agent.
    action_composer:
        Template :class:`~wrappers.actions.composer.ActionComposer`, cloned per agent for
        denormalizing and constraining actions.  Applied to every trainable
        agent's output.
    action_repeat:
        Number of env steps per agent decision.
    hooks:
        Optional list of :class:`~training.hooks.TrainingHook` callbacks.
    render:
        Whether to call ``env.render()`` each step.
    focal_agent_id:
        Agent ID used for episode-level logging (reward, outcome).  Defaults
        to the first element of *trainable_ids*.
    run_id:
        Stable run identifier used in per-agent dataset transition records.
    """

    def __init__(
        self,
        env: Any,
        agent: MAPPOAgent,
        trainable_ids: List[str],
        other_agents: Dict[str, Any],
        obs_composers: Dict[str, ObservationComposer],
        reward_composers: Dict[str, RewardComposer],
        action_composer: ActionComposer,
        action_repeat: int = 1,
        hooks: Optional[List[TrainingHook]] = None,
        render: bool = False,
        focal_agent_id: Optional[str] = None,
        run_id: str = "run",
        reward_mode: str = "individual",
        team_reward_reduction: str = "mean",
    ) -> None:
        self.env = env
        self.agent = agent
        self.trainable_ids = list(trainable_ids)
        self.other_agents = other_agents
        self.obs_composers = obs_composers
        self.reward_composers = reward_composers
        self.action_composer = action_composer
        self.action_composers = {aid: copy.deepcopy(action_composer) for aid in self.trainable_ids}
        self.action_repeat = max(1, int(action_repeat))
        self.hooks = hooks or []
        self._environment_steps = 0
        self._physics_steps = 0
        self._agent_steps = 0
        self._updates = 0
        self._transition_hooks = transition_record_hooks(self.hooks)
        self.render = render
        self.focal_id = focal_agent_id or (trainable_ids[0] if trainable_ids else "")
        self.run_id = run_id
        self.reward_mode = str(reward_mode).strip().lower()
        self.team_reward_reduction = str(team_reward_reduction).strip().lower()
        self.team_return_mode = getattr(agent, "team_return_mode", "per_agent")
        self._has_team_rewards = validate_team_reward_composers(
            reward_composers, trainable_ids=self.trainable_ids,
            opponent_ids=list(other_agents), reward_mode=self.reward_mode,
            critic_mode=getattr(agent, "critic_mode", "agent_conditioned"),
            team_return_mode=self.team_return_mode, action_repeat=self.action_repeat,
        )
        if self.team_return_mode == "joint" and self.action_repeat != 1:
            raise ValueError("Joint team returns currently require action_repeat=1")
        if self.reward_mode not in {"individual", "team_shared"}:
            raise ValueError(
                "MAPPO reward_mode must be 'individual' or 'team_shared', "
                f"got {self.reward_mode!r}."
            )
        if self.team_reward_reduction not in {"mean", "sum"}:
            raise ValueError(
                "MAPPO team_reward_reduction must be 'mean' or 'sum', "
                f"got {self.team_reward_reduction!r}."
            )
        agent_ids = getattr(agent, "agent_ids", None)
        if agent_ids is not None and list(agent_ids) != self.trainable_ids:
            raise ValueError(
                "MARLTrainer trainable_ids must exactly match the MAPPO agent ID order."
            )
        for field in ("reward_mode", "team_reward_reduction"):
            agent_value = getattr(agent, field, None)
            trainer_value = getattr(self, field)
            if agent_value is not None and agent_value != trainer_value:
                raise ValueError(
                    f"MARLTrainer {field}={trainer_value!r} does not match "
                    f"MAPPOAgent {field}={agent_value!r}."
                )

        self.task = RaceTask(
            env, policy_agents=self.trainable_ids, fixed_controllers=other_agents,
            obs_composers=obs_composers, reward_composers=reward_composers,
            action_composers=self.action_composers, action_repeat=self.action_repeat,
            team_reward_agent_id=self.focal_id if self._has_team_rewards else None,
        )

    def _episode_id(self, episode: int) -> str:
        return f"{self.run_id}_ep{episode:06d}"

    def _map_id(self) -> Optional[str]:
        return getattr(self.env, "_map_bundle_active", None) or getattr(
            self.env, "map_name", None
        )

    def _spawn_id(self, agent_id: str) -> Optional[str]:
        spawn_manager = getattr(self.env, "_spawn_manager", None)
        if spawn_manager is None:
            return None
        metadata = getattr(spawn_manager, "last_spawn_metadata", {}) or {}
        spawn_ids = metadata.get("spawn_ids", {})
        return (spawn_ids.get(agent_id) or metadata.get("spawn_id") or
                getattr(spawn_manager, "last_spawn_mapping", {}).get(agent_id))

    def _learning_rewards(
        self,
        individual_rewards: Dict[str, float],
    ) -> Dict[str, float]:
        """Map factual per-agent rewards to the configured learning signal.

        Team means always use the configured team size as denominator.  A
        teammate that already terminated therefore contributes zero instead
        of silently changing reward scale as the active set shrinks.
        """
        return map_mappo_learning_rewards(
            individual_rewards,
            trainable_ids=self.trainable_ids,
            reward_mode=self.reward_mode,
            team_reward_reduction=self.team_reward_reduction,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, n_episodes: int = 0, *, total_steps: Optional[int] = None) -> None:
        """Train to an episode count or exact joint environment-decision budget."""
        for _ in self.iter_train(n_episodes, total_steps=total_steps):
            pass

    def train_parallel(self, scenario, scenario_dir, num_envs, n_episodes=0, *, total_steps=None):
        from training.parallel_mappo import train_parallel
        train_parallel(self, scenario, scenario_dir, num_envs, n_episodes, total_steps=total_steps)

    def _reset_task(self):
        return self.task.reset()

    def _step_task(self, actions, on_physics_step):
        return self.task.step(actions, on_physics_step=on_physics_step)

    def _should_stop(self):
        return bool(getattr(self.agent, "should_stop", False)) or any(
            getattr(hook, "should_stop", False) for hook in self.hooks)

    def iter_train(self, n_episodes: int, *, parallel: bool = False,
                   total_steps: Optional[int] = None):
        """Shared race loop; a budget cut bootstraps without ending the race."""
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
            spawn_context = capture_spawn_context(self.env, facts.agents)
            physics_steps = 0
            start_policy = getattr(self.agent, "policy_version", self._updates)
            finite_race = getattr(getattr(self.env, "lifecycle", None), "finish_on_laps", True)
            team_components = {}

            def on_physics_step(substep):
                nonlocal physics_steps
                physics_steps += 1
                self._physics_steps += 1
                if parallel:
                    self.agent.physics_steps_collected = self._physics_steps
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
                if parallel:
                    actions_norm, log_probs, values, raw_actions = yield (
                        "act", (active_trainable_ids, stacked_observations, global_state)
                    )
                    self.agent.last_raw_actions = raw_actions
                else:
                    if active_trainable_ids:
                        actions_norm, log_probs = self.agent.act_batch(
                            active_trainable_ids, stacked_observations,
                        )
                    else:
                        actions_norm, log_probs = {}, {}
                    values = self.agent.evaluate_states(global_state, active_trainable_ids)
                task_step = self._step_task(actions_norm, on_physics_step=on_physics_step)
                info_dict = task_step.after.infos
                actions_phys = {aid: decision.action_physical for aid, decision in task_step.decisions.items()}
                accumulated_individual_rewards = {
                    aid: decision.individual_reward for aid, decision in task_step.decisions.items()
                }
                accumulated_learning_rewards = self._learning_rewards(accumulated_individual_rewards)
                for aid in accumulated_learning_rewards:
                    accumulated_learning_rewards[aid] += task_step.team_reward
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
                    **({"raw_actions": self.agent.last_raw_actions}
                       if hasattr(self.agent, "last_raw_actions") else {}),
                )
                if self.team_return_mode == "joint" and ordered_ids:
                    self.agent.store_team_step(
                        ordered_ids, reward=team_step_reward,
                        value=values[ordered_ids[0]],
                        terminal=not any(aid in getattr(self.env, "agents", []) for aid in self.trainable_ids),
                    )

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
                                self.env,
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

                if parallel:
                    yield "step", next_global_state

                # --- Trigger update when any buffer is full or episode ends ---
                if self.agent.any_buffer_full() or episode_done or budget_done:
                    if parallel:
                        next_values = yield "value", next_global_state
                        update_metrics = self.agent.finish_fragment(next_values)
                    else:
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
                    timestep=float(getattr(self.env, "timestep", .01)), finite_race=finite_race),
                "run_id": self.run_id, "environment_id": 0, "episode_id": episode_id,
                "environment_episode": episode, "map_id": map_id,
                "environment_seed": getattr(self.env, "seed", None),
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
