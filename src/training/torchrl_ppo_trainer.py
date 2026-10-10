from __future__ import annotations

import time

import torch
from torchrl.envs import step_mdp
from torchrl.envs.utils import ExplorationType, set_exploration_type

from adapters import RaceGymEnv
from adapters.torchrl import RaceTorchRLEnv
from env.types import TransitionRecord
from metrics.outcomes import determine_outcome
from training.ppo_collector import PPOTrainerBase
from training.transition_records import transition_lifecycle_fields


class TorchRLPPOTrainer(PPOTrainerBase):
    def train_parallel(self, scenario, scenario_dir, num_envs, n_episodes=0, *, total_steps=None):
        from training.parallel_ppo import train_parallel
        return train_parallel(self, scenario, scenario_dir, num_envs, n_episodes, total_steps=total_steps)

    def train(self, n_episodes=0, *, total_steps=None):
        if total_steps is not None and (
            isinstance(total_steps, bool) or not isinstance(total_steps, int) or total_steps <= 0
        ):
            raise ValueError("total_steps must be a positive integer")
        if total_steps is None and (isinstance(n_episodes, bool) or not isinstance(n_episodes, int) or n_episodes < 0):
            raise ValueError("n_episodes must be a nonnegative integer")
        gym_env = RaceGymEnv(self.task)
        if not self.render:
            gym_env.render_mode = None
        wrapped = RaceTorchRLEnv(gym_env, device="cpu")
        episode, self.collected_steps, updates = 0, 0, 0
        started = time.perf_counter()
        pending, pending_steps = [], 0
        metrics = {}
        self._set_training_progress(0, total_steps or n_episodes)

        def report_update(metrics, update_started):
            nonlocal updates
            if not metrics:
                return
            updates += 1
            metrics.update({
                "train/environment_steps": self.collected_steps, "train/updates": updates,
                "perf/update_seconds": time.perf_counter() - update_started,
                "perf/end_to_end_env_steps_per_second": self.collected_steps / max(time.perf_counter() - started, 1e-9),
            })
            for hook in self.hooks:
                hook.on_update(metrics)

        try:
            while (self.collected_steps < total_steps if total_steps is not None else episode < n_episodes):
                if self._should_stop():
                    break
                current = wrapped.reset()
                episode_reward, step_idx = 0.0, 0
                metrics = {}
                episode_id, map_id, spawn_id = self._episode_id(episode), self._map_id(), self._spawn_id()
                for hook in self.hooks:
                    callback = getattr(hook, "on_episode_start", None)
                    if callback is not None:
                        callback(dict(episode_id=episode_id, map_id=map_id,
                                      physics=gym_env.snapshot.infos[self.rl_agent_id].get("physics")))

                def on_decision():
                    result = gym_env.last_step
                    decision = result.decisions[self.rl_agent_id]
                    if self._transition_hooks:
                        record = TransitionRecord(
                            obs=decision.observation, action_norm=decision.action_normalized,
                            action_phys=decision.action_physical, reward=decision.individual_reward,
                            reward_components=dict(decision.reward_components), next_obs=decision.next_observation,
                            terminated=decision.terminated, truncated=decision.truncated, info=dict(decision.info),
                            global_state=result.before.global_state.vector.copy(), map_id=map_id, spawn_id=spawn_id,
                            episode_id=episode_id, step_idx=step_idx, agent_id=self.rl_agent_id,
                            **transition_lifecycle_fields(decision.info, global_state=result.after.global_state),
                        )
                        for hook in self._transition_hooks:
                            hook.on_step(record)

                done = False
                while not done and not self._should_stop():
                    remaining = self.agent.n_steps - pending_steps
                    if total_steps is not None:
                        remaining = min(remaining, total_steps - self.collected_steps)
                    if remaining <= 0:
                        break
                    if self._transition_hooks:
                        remaining = min(remaining, 1)
                    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
                        rollout = wrapped.rollout(
                            remaining, self.agent.policy,
                            auto_reset=False, tensordict=current, auto_cast_to_device=True,
                            break_when_any_done=True, return_contiguous=True,
                        )
                    if self._transition_hooks:
                        on_decision()
                    # TorchRL's rollout callback omits boundary steps. Account
                    # from the returned batch, including its final transition.
                    episode_reward += float(rollout["next", "reward"].sum())
                    step_idx += rollout.numel()
                    self.collected_steps += rollout.numel()
                    pending.append(rollout)
                    pending_steps += rollout.numel()
                    current = step_mdp(rollout[-1])
                    done = bool(rollout["next", "done"][-1].item())
                    budget_done = total_steps is not None and self.collected_steps >= total_steps
                    if pending_steps >= self.agent.n_steps or budget_done or (done and total_steps is None):
                        if total_steps is not None:
                            self._set_training_progress(self.collected_steps, total_steps)
                        update_started = time.perf_counter()
                        metrics = self.agent.update(torch.cat(pending, dim=0))
                        pending, pending_steps = [], 0
                        report_update(metrics, update_started)
                    if budget_done:
                        break
                if not done:
                    break
                decision = gym_env.last_step.decisions[self.rl_agent_id]
                info = dict(decision.info)
                info["outcome"] = determine_outcome(info, truncated=decision.truncated).value
                info.setdefault("map_bundle", map_id)
                episode_metrics = dict(metrics, episode_steps=step_idx)
                lap_steps = info.get("lap_time_steps")
                episode_metrics["lap_time_s"] = float(lap_steps) * self.task.timestep if lap_steps is not None else None
                if total_steps is None:
                    self._set_training_progress(episode + 1, n_episodes)
                for hook in self.hooks:
                    hook.on_episode_end(episode, episode_reward, info, episode_metrics)
                episode += 1
                if total_steps is not None and self.collected_steps >= total_steps:
                    break
            if not self._should_stop():
                update_started = time.perf_counter()
                report_update(self.agent.flush_pending_update(), update_started)
            for hook in self.hooks:
                hook.on_training_end()
        finally:
            wrapped.close()
