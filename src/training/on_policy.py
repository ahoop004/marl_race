"""Direct TorchRL collection and a small shared PPO/MAPPO update loop."""
import time

import torch
from torchrl.collectors import Collector

from adapters.native_torchrl import NativeRaceTorchRLEnv
from training.contracts import validate_team_reward_composers
from training.race_reporting import RaceTrainingReports


class _CollectionBoundary:
    """The supported Collector interruptor protocol, evaluated after each step."""
    def __init__(self, trainer):
        self.trainer = trainer

    def collection_stopped(self):
        t = self.trainer
        return (t.budget_reached() or t._should_stop()
                or t._pending_learning_steps >= t.agent.n_steps
                or t.env.last_step is None)  # Collector has just auto-reset a race.


class OnPolicyTrainer:
    def __init__(self, task, agent, *, hooks=None, render=False, run_id="run",
                 focal_agent_id=None, algorithm=None):
        self.task, self.agent = task, agent
        self.algorithm = algorithm or ("mappo" if hasattr(agent, "global_state_dim") else "ppo")
        if self.algorithm not in {"ppo", "mappo"}:
            raise ValueError(f"Unsupported on-policy algorithm: {self.algorithm!r}")
        self.hooks, self.render, self.run_id = hooks or [], render, run_id
        self.trainable_ids = list(task.possible_agents)
        self.focal_id = focal_agent_id or self.trainable_ids[0]
        if self.focal_id not in self.trainable_ids:
            raise ValueError("Focal agent must be a policy agent")
        if self.algorithm == "ppo":
            if len(self.trainable_ids) != 1 or task.team_reward_agent_id is not None:
                raise ValueError("PPO requires exactly one learner and individual rewards")
        else:
            if list(agent.agent_ids) != self.trainable_ids:
                raise ValueError("Task policy agents must exactly match MAPPO agent ID order")
            validate_team_reward_composers(
                task.reward_composers, trainable_ids=self.trainable_ids,
                opponent_ids=list(task.fixed_controllers), reward_mode=agent.reward_mode,
                critic_mode=agent.critic_mode, team_return_mode=agent.team_return_mode,
                action_repeat=task.action_repeat)
            if task.action_repeat != 1:
                raise ValueError("Joint team returns require action_repeat=1")
        self._environment_steps = self._physics_steps = self._agent_steps = self._updates = 0
        self._pending_learning_steps = 0
        self.total_steps = self.n_episodes = None
        self.env = None
        self.reports = RaceTrainingReports(self)
        self._optimizer_steps = 0
        self._resume_state = None
        self.checkpoint_ready = False
        for hook in self.hooks:
            bind = getattr(hook, "bind_trainer", None)
            if bind:
                bind(self)

    def checkpoint_progress(self):
        return {"environment_steps": self._environment_steps, "agent_steps": self._agent_steps,
                "physics_steps": self._physics_steps, "completed_episodes": self.reports.completed,
                "updates": self._updates, "optimizer_steps": self._optimizer_steps,
                "next_episode": self.reports.next_episode}

    def checkpoint_state(self):
        """Recovery at a completed update; the next collection starts a new race.

        Simulator/controller state and partial episodes are deliberately not
        serialized. Map/spawn/friction generators restart at recovery_seed.
        This is deterministic recovery, not an exact next-transition resume.
        """
        from agents.common.checkpoints import RESUME_BOUNDARY, capture_rng, optimization_contract
        if not self.checkpoint_ready or getattr(self.agent, "_pending_batches", []):
            raise ValueError("Resumable checkpoints require a completed update with no pending trajectories")
        completed, total = ((self._environment_steps, self.total_steps) if self.total_steps is not None
                            else (self.reports.completed, self.n_episodes))
        return {"boundary": RESUME_BOUNDARY, "optimizer": self.agent.optimizer.state_dict(),
                "optimization_contract": optimization_contract(self.agent),
                "schedule_progress": completed / max(total, 1), "progress": self.checkpoint_progress(),
                "budget": {"total_steps": self.total_steps, "n_episodes": self.n_episodes},
                "rng": capture_rng(), "recovery_seed": getattr(self.task.env, "seed", None) or 0,
                "hooks": {type(hook).__name__: hook.state_dict() for hook in self.hooks
                          if hasattr(hook, "state_dict")},
                "recovery": {"discard_partial_episode": True, "environment_schedules": "restart_at_seed"}}

    def restore_training_state(self, state):
        self._resume_state = state
        progress = state["progress"]
        self._environment_steps, self._agent_steps, self._physics_steps = (
            progress["environment_steps"], progress["agent_steps"], progress["physics_steps"])
        self._updates, self._optimizer_steps = progress["updates"], progress["optimizer_steps"]
        self.reports.completed = progress["completed_episodes"]
        self.reports.next_episode = progress["next_episode"]
        for hook in self.hooks:
            if hasattr(hook, "load_state_dict"):
                if type(hook).__name__ not in state["hooks"]:
                    raise ValueError(f"Resume checkpoint missing state for {type(hook).__name__}")
                hook._environment_steps, hook._policy_version = self._environment_steps, self._updates
                hook.load_state_dict(state["hooks"][type(hook).__name__])

    @property
    def collected_steps(self):
        return self._environment_steps

    def _should_stop(self):
        return bool(getattr(self.agent, "should_stop", False)) or any(
            getattr(hook, "should_stop", False) for hook in self.hooks)

    def budget_reached(self):
        if self.total_steps is not None:
            return self._environment_steps >= self.total_steps
        return self.n_episodes is not None and self.reports.completed >= self.n_episodes

    def _set_progress(self):
        setter = getattr(self.agent, "set_training_progress", None)
        if setter:
            completed, total = ((self._environment_steps, self.total_steps) if self.total_steps is not None
                                else (self.reports.completed, self.n_episodes))
            setter(completed / max(total, 1))

    def train(self, n_episodes=0, *, total_steps=None):
        if total_steps is not None and (type(total_steps) is not int or total_steps <= 0):
            raise ValueError("total_steps must be a positive integer")
        if total_steps is None and (type(n_episodes) is not int or n_episodes < 0):
            raise ValueError("n_episodes must be a nonnegative integer")
        if self._resume_state is not None:
            from utils.torch_io import validate_checkpoint_compatibility
            validate_checkpoint_compatibility(self._resume_state["budget"], {
                "total_steps": total_steps, "n_episodes": n_episodes})
        for hook in self.hooks:
            bind = getattr(hook, "bind_trainer", None)
            if bind:
                bind(self)
        self.total_steps, self.n_episodes = total_steps, n_episodes
        self.env = NativeRaceTorchRLEnv(
            self.task, reward_mode=getattr(self.agent, "reward_mode", "individual"),
            team_reward_reduction=getattr(self.agent, "team_reward_reduction", "mean"))
        self.env.render_mode = "human" if self.render else None
        self.env.on_reset = self.reports.reset
        self.env.on_decision = self.reports.decision
        self.env.on_physics_step = self.reports.physics
        pending = []
        started = collection_started = time.perf_counter()
        metrics = {}
        collector = None
        self._set_progress()

        def report_update(updated, update_started):
            nonlocal collection_started
            if updated:
                self._updates += 1
                self._optimizer_steps += int(updated["train/optimizer_steps"])
                updated.update({
                    "train/environment_steps": self._environment_steps,
                    "train/physics_steps": self._physics_steps, "train/agent_steps": self._agent_steps,
                    "train/updates": self._updates,
                    "perf/update_seconds": time.perf_counter() - update_started,
                    "perf/collection_seconds": update_started - collection_started,
                    "perf/elapsed_seconds": time.perf_counter() - started,
                    "perf/end_to_end_env_steps_per_second": self._environment_steps / max(time.perf_counter() - started, 1e-9)})
                for hook in self.hooks:
                    hook.on_update(updated)
                collection_started = time.perf_counter()
            return updated

        def checkpoint_boundary():
            self.checkpoint_ready = True
            for hook in self.hooks:
                callback = getattr(hook, "on_checkpoint_boundary", None)
                if callback:
                    callback()

        def optimize(data):
            self._set_progress()
            update_started = time.perf_counter()
            return report_update(self.agent.update(data), update_started)

        try:
            require_source = getattr(self.agent, "_require_lora_source", None)
            if require_source:
                require_source()
            if not self.budget_reached():
                # An unbounded collector plus its interruption protocol avoids
                # 0.14's rounding of non-divisible total_frames. No private
                # collector state, manual stepping, or artificial truncation.
                if self._resume_state is not None:
                    self.env.set_seed(self._resume_state["recovery_seed"])
                collector = Collector(
                    self.env, self.agent.collection_policy, backend="direct",
                    frames_per_batch=self.agent.n_steps, total_frames=-1,
                    policy_device=self.agent.device, env_device="cpu", storing_device="cpu",
                    reset_at_each_iter=False, set_truncated=False, use_buffers=True,
                    auto_register_policy_transforms=False,
                    # Buffered interruption pads unused rows with traj_ids=-1.
                    postproc=lambda batch: batch[batch["collector", "traj_ids"] >= 0],
                    interruptor=_CollectionBoundary(self))
                if self._resume_state is not None:
                    from agents.common.checkpoints import restore_rng
                    restore_rng(self._resume_state["rng"])
                for data in collector:
                    self.checkpoint_ready = False
                    updated = False
                    pending.append(data)
                    boundary = self.env.last_step is None
                    ready = (self._pending_learning_steps >= self.agent.n_steps or self.budget_reached()
                             or (boundary and (self.algorithm == "mappo" or total_steps is None)))
                    if ready:
                        metrics = optimize(torch.cat(pending, dim=0))
                        pending, self._pending_learning_steps = [], 0
                        updated = bool(metrics)
                    self.reports.dispatch(metrics)
                    if updated:
                        checkpoint_boundary()
                    if self.budget_reached() or self._should_stop():
                        break
                if not self._should_stop() and pending:
                    metrics = optimize(torch.cat(pending, dim=0))
                    pending, self._pending_learning_steps = [], 0
                    if metrics:
                        checkpoint_boundary()
                flush = getattr(self.agent, "flush_pending_update", None)
                if not self._should_stop() and flush:
                    self._set_progress()
                    update_started = time.perf_counter()
                    if report_update(flush(), update_started):
                        checkpoint_boundary()
            for hook in self.hooks:
                hook.on_training_end()
        finally:
            if collector is not None:
                collector.shutdown()
            else:
                self.env.close()
