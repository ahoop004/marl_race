"""Task setup and the scheduled PPO worker collection loop."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from env.types import TransitionRecord
from metrics.outcomes import determine_outcome
from training.hooks import TrainingHook, transition_record_hooks
from training.reward_context import transition_lifecycle_fields
from tasks import RaceTask, TaskSnapshot


class PPOTrainerBase:
    """Task ownership, hook metadata and global training progress."""

    def __init__(
        self, task: RaceTask, agent: Any, *,
        hooks: Optional[List[TrainingHook]] = None, render: bool = False,
        run_id: str = "run", spawn_plan_fn: Optional[Callable] = None,
    ) -> None:
        if len(task.possible_agents) != 1:
            raise ValueError("PPO requires exactly one policy agent")
        if task.team_reward_agent_id is not None:
            raise ValueError("Team race rewards require MAPPO joint team returns")
        self.task = task
        self.env = task.env
        self.rl_agent_id = task.possible_agents[0]
        self.agent = agent
        self.hooks = hooks or []
        self._transition_hooks = transition_record_hooks(self.hooks)
        self.render = render
        self.run_id = run_id
        self.spawn_plan_fn = spawn_plan_fn
        self.collected_steps = 0

    def _set_training_progress(self, completed: int, total: int) -> None:
        # Remote collectors own no optimizer; the parent uses global progress.
        setter = getattr(self.agent, "set_training_progress", None)
        if setter is not None:
            setter(completed / max(total, 1))

    def _episode_id(self, episode: int) -> str:
        return f"{self.run_id}_ep{episode:06d}"

    def _map_id(self) -> Optional[str]:
        return getattr(self.env, "_map_bundle_active", None) or getattr(
            self.env, "map_name", None
        )

    def _spawn_id(self) -> Optional[str]:
        sm = getattr(self.env, "_spawn_manager", None)
        if sm is None:
            return None
        meta = getattr(sm, "last_spawn_metadata", {}) or {}
        spawn_ids = meta.get("spawn_ids", {})
        return spawn_ids.get(self.rl_agent_id) or meta.get("spawn_id")

    def _reset_env(self) -> TaskSnapshot:
        """Reset env, injecting a curriculum spawn plan when one is available."""
        spawn_plan = self.spawn_plan_fn() if self.spawn_plan_fn is not None else None
        options = {"spawn_plan": spawn_plan} if spawn_plan is not None else None
        return self.task.reset(options=options)

    def _on_physics_step(self, substep) -> None:
        if self.render:
            try:
                self.task.render()
            except Exception:
                pass

    def _should_stop(self) -> bool:
        return bool(getattr(self.agent, "should_stop", False)) or any(
            getattr(hook, "should_stop", False) for hook in self.hooks)

    def _flush_pending_update(self) -> None:
        flush = getattr(self.agent, "flush_pending_update", None)
        metrics = flush() if flush is not None else {}
        if metrics:
            metrics["train/environment_steps"] = self.collected_steps
            for hook in self.hooks:
                hook.on_update(metrics)


class PPOCollector(PPOTrainerBase):
    """Collect task decisions while the parent owns inference and updates."""

    def _policy_request(self, observation):
        action, log_prob, value, raw = yield "act", observation
        return action, log_prob, value, raw

    def iter_train(self, n_episodes: int = 0, *, total_steps: Optional[int] = None):
        """Collect TensorDict fragments under the parent's frozen policy."""
        if total_steps is not None and total_steps <= 0:
            raise ValueError("total_steps must be positive")
        self._set_training_progress(0, total_steps or n_episodes)
        collected = 0
        episode = 0
        self.agent.buffer.clear()
        while (collected < total_steps if total_steps is not None else episode < n_episodes):
            snapshot = self._reset_env()
            info_dict = snapshot.infos
            if total_steps is None:
                self.agent.buffer.clear()
            obs = snapshot.observations[self.rl_agent_id]
            done = False
            episode_reward = 0.0
            episode_truncated = False
            update_metrics: Dict = {}
            last_info: Dict = {}
            step_idx = 0
            episode_id = self._episode_id(episode)
            map_id = self._map_id()
            spawn_id = self._spawn_id()
            for hook in self.hooks:
                callback = getattr(hook, "on_episode_start", None)
                if callback is not None:
                    callback(dict(episode_id=episode_id, map_id=map_id,
                        physics=info_dict.get(self.rl_agent_id, {}).get("physics")))

            while not done:
                action_norm, log_prob, value, raw_action = yield from self._policy_request(obs)
                task_step = self.task.step(
                    {self.rl_agent_id: action_norm}, on_physics_step=self._on_physics_step,
                )
                decision = task_step.decisions[self.rl_agent_id]
                action_phys = decision.action_physical
                reward = decision.individual_reward
                rl_term, rl_trunc = decision.terminated, decision.truncated
                done = rl_term or rl_trunc
                episode_truncated = bool(rl_trunc) if done else episode_truncated
                post_step_global_snapshot = task_step.after.global_state
                global_state = task_step.before.global_state.vector.copy() if self._transition_hooks else None
                last_info = dict(decision.info)
                episode_reward += reward
                next_obs = decision.next_observation

                # --- Emit transition record for dataset hooks ---
                if self._transition_hooks:
                    record = TransitionRecord(
                        obs=obs,
                        action_norm=action_norm,
                        action_phys=action_phys,
                        reward=reward,
                        reward_components=dict(decision.reward_components),
                        next_obs=next_obs,
                        terminated=rl_term if done else False,
                        truncated=rl_trunc if done else False,
                        info=last_info,
                        global_state=global_state,
                        map_id=map_id,
                        spawn_id=spawn_id,
                        episode_id=episode_id,
                        step_idx=step_idx,
                        agent_id=self.rl_agent_id,
                        **transition_lifecycle_fields(
                            self.env,
                            last_info,
                            global_state=post_step_global_snapshot,
                        ),
                    )
                    for hook in self._transition_hooks:
                        hook.on_step(record)
                step_idx += 1

                collected += 1
                self.collected_steps = collected
                budget_done = total_steps is not None and collected >= total_steps
                self.agent.buffer.add(
                    obs, action_norm, reward, log_prob, value,
                    terminated=rl_term, truncated=rl_trunc,
                    raw_action=raw_action, next_observation=next_obs,
                )

                if self.agent.buffer.is_full() or budget_done or (done and total_steps is None):
                    # The learner owns GAE and bootstraps from stored next observations.
                    reply = yield "rollout", self.agent.pack_rollout()
                    update_metrics = self.agent.apply_reply(reply)
                    self.agent.buffer.clear()
                    if update_metrics:
                        update_metrics["train/environment_steps"] = collected
                        for hook in self.hooks:
                            hook.on_update(update_metrics)

                obs = next_obs
                if budget_done or self._should_stop():
                    break

            # Exhausting the training budget is not an environment terminal.
            if not done:
                break
            outcome = determine_outcome(last_info, truncated=episode_truncated)
            last_info["outcome"] = outcome.value
            last_info.setdefault("map_bundle", map_id)
            update_metrics = dict(update_metrics)
            update_metrics["episode_steps"] = step_idx
            # Lifecycle timing persists after the crossing and counts physics
            # steps, not policy decisions (so do not multiply by action_repeat).
            lap_time_steps = last_info.get("lap_time_steps")
            timestep = getattr(self.env, "timestep", None)
            update_metrics["lap_time_s"] = (
                float(lap_time_steps) * float(timestep)
                if lap_time_steps is not None and timestep is not None else None
            )

            if total_steps is None:
                self._set_training_progress(episode + 1, n_episodes)
            for hook in self.hooks:
                hook.on_episode_end(episode, episode_reward, last_info, update_metrics)
            episode += 1
            if self._should_stop():
                break

        # A passed evaluation gate freezes the evaluated weights exactly.
        if not self._should_stop():
            self._flush_pending_update()
        for hook in self.hooks:
            hook.on_training_end()

