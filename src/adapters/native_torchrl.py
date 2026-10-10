"""Native TorchRL translation of RaceTask with fixed learner slots."""
from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.data import Bounded, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

from adapters.rewards import RewardMapping
from tasks import RaceTaskProtocol, TaskSnapshot, TaskStep


class NativeRaceTorchRLEnv(EnvBase):
    """One unbatched race; agents are a nested batch, never environment batches.

    Vector observations are right-padded to the largest learner width; the
    task specification retains individual widths for per-agent LoRA routing.
    Root done flags reset the physical race. Agent flags retain individual
    boundaries as observations: registering them in done_spec would advance
    Collector trajectory IDs at retirement. ``learning`` describes the existing
    joint-team return endpoint:
    no learner can act, including at a timeout (no team bootstrap). It is an
    observation, not a reset signal. Current-state ``agents.active`` selects
    real actor samples, including their final actions; fixed-only steps have
    no samples. Inactive slots retain their last observation and boundary flags.
    """

    @classmethod
    def from_scenario(cls, scenario, *, scenario_dir=None, mode="train",
                      render_mode=None, **options):
        from core.task_builder import create_race_task

        task = create_race_task(scenario, scenario_dir=scenario_dir, mode=mode,
                                render_mode=render_mode)
        try:
            return cls(task, **options)
        except BaseException:
            task.close()
            raise

    def __init__(self, task: RaceTaskProtocol, *, reward_mode="individual",
                 team_reward_reduction="mean", device="cpu"):
        super().__init__(device=device, batch_size=(), run_type_checks=True)
        self.task = task
        self.agent_ids = tuple(task.possible_agents)
        self.physical_agent_ids = tuple(task.physical_agents)
        if not self.agent_ids:
            raise ValueError("The native environment requires at least one policy agent")
        widths = {task.observation_space(aid).shape for aid in self.agent_ids}
        if any(len(width) != 1 for width in widths):
            raise ValueError("The agents group requires vector observations")
        for aid in self.agent_ids:
            spec = task.action_space(aid)
            if spec.shape != (2,) or not (np.all(spec.low == -1) and np.all(spec.high == 1)):
                raise ValueError("Policy actions must be normalized float32 vectors of shape (2,)")
        self.group_map = {"agents": list(self.agent_ids)}
        self.reward_mapping = RewardMapping(reward_mode, team_reward_reduction)
        self.render_mode = task.render_mode
        self.snapshot: TaskSnapshot | None = None
        self.last_step: TaskStep | None = None
        self.on_physics_step = None
        self.on_reset = None
        self.on_decision = None
        self._pending_seed = None
        n, p = len(self.agent_ids), len(self.physical_agent_ids)
        self._observations = np.zeros((n, max(width[0] for width in widths)), dtype=np.float32)
        self._terminated = np.zeros((n, 1), dtype=bool)
        self._truncated = np.zeros((n, 1), dtype=bool)

        def boolean(shape):
            return Categorical(2, shape=shape, dtype=torch.bool, device=self.device)

        def real(shape):
            return Unbounded(shape=shape, dtype=torch.float32, device=self.device)

        def count(shape):
            return Unbounded(shape=shape, dtype=torch.int64, device=self.device)

        self.observation_spec = Composite(
            state=real(task.state_space().shape),
            agents=Composite(observation=real(self._observations.shape),
                             index=count((n,)), active=boolean((n, 1)),
                             **{key: boolean((n, 1)) for key in (
                                 "done", "terminated", "truncated")}, shape=(n,)),
            physical=Composite(**{key: boolean((p, 1)) for key in (
                "active", "controlled", "trainable", "finished", "crashed",
                "truncated", "task_complete")}, shape=(p,)),
            learning=Composite(done=boolean((1,)), terminated=boolean((1,)),
                               truncated=boolean((1,)), valid=boolean((1,))),
            race_episode_done=boolean((1,)),
            decision_steps=count((1,)), physics_steps=count((1,)),
            decision_physics_steps=count((1,)), agent_steps=count((1,)),
            elapsed_seconds=real((1,)), device=self.device,
        )
        self.action_spec = Composite(agents=Composite(
            action=Bounded(-1, 1, shape=(n, 2), dtype=torch.float32, device=self.device),
            shape=(n,), device=self.device), device=self.device)
        self.reward_spec = Composite(
            agents=Composite(reward=real((n, 1)), individual_reward=real((n, 1)),
                             shape=(n,), device=self.device),
            team_reward=real((1,)), shared_bonus=real((1,)), device=self.device,
        )
        self.done_spec = Composite(
            **{key: boolean((1,)) for key in ("done", "terminated", "truncated")},
            device=self.device,
        )
        self.is_closed = False

    def _tensor(self, value, dtype):
        # Copy task arrays: TensorDict transitions must survive later steps/resets.
        return torch.tensor(value, dtype=dtype, device=self.device)

    def _set_seed(self, seed: int | None):
        # EnvBase.set_seed must not consume a task reset or process RNG draws.
        self._pending_seed = seed

    def _reset(self, tensordict=None, *, seed=None, options=None):
        if self.is_closed:
            raise RuntimeError("The environment is closed")
        seed = self._pending_seed if seed is None else seed
        snapshot = self.task.reset(seed=seed, options=options)
        if tuple(snapshot.agents) != self.agent_ids:
            raise ValueError("Reset must activate every configured policy agent in order")
        self._pending_seed = None
        self._observations.fill(0)
        self._terminated.fill(False)
        self._truncated.fill(False)
        self.snapshot, self.last_step = snapshot, None
        for i, aid in enumerate(self.agent_ids):
            self._observations[i, :len(snapshot.observations[aid])] = snapshot.observations[aid]
        output = self._output(snapshot)
        if self.on_reset is not None:
            self.on_reset(snapshot)
        if self.render_mode == "human":
            self.task.render()
        return output

    def _on_physics_step(self, substep):
        if self.on_physics_step is not None:
            self.on_physics_step(substep)
        if self.render_mode == "human":
            self.task.render()

    def _step(self, tensordict: TensorDictBase):
        if self.is_closed or self.snapshot is None or self.snapshot.episode_done:
            raise RuntimeError("Reset an open environment before stepping a new race")
        action = tensordict.get(("agents", "action"), None)
        if not isinstance(action, torch.Tensor) or action.shape != (len(self.agent_ids), 2):
            raise ValueError("agents.action must have shape (number of policy agents, 2)")
        if action.dtype != torch.float32:
            raise ValueError("agents.action must have dtype float32")
        active = set(self.snapshot.agents)
        actions = {aid: action[i].detach().cpu().numpy().copy()
                   for i, aid in enumerate(self.agent_ids) if aid in active}
        # RaceTask validates all active actions before any composer/controller runs.
        result = self.task.step(actions, on_physics_step=self._on_physics_step)
        for i, aid in enumerate(self.agent_ids):
            decision = result.decisions.get(aid)
            if decision is not None:
                self._observations[i, :len(decision.next_observation)] = decision.next_observation
                self._terminated[i] = decision.terminated
                self._truncated[i] = decision.truncated
        self.snapshot, self.last_step = result.after, result
        output = self._output(result.after, result)
        if self.on_decision is not None:
            self.on_decision(result)
        return output

    @staticmethod
    def _physical_boundary(result):
        if result is None or not result.after.episode_done:
            return False, False
        facts = result.substeps[-1].facts
        # Ignore flags on cars retired before this decision. A timeout of the
        # final active opponent is still a root truncation during fixed-only play.
        active = result.before.active_physical_agents
        truncated = any(facts.truncations.get(aid, False) for aid in active)
        terminated = any(facts.terminations.get(aid, False) for aid in active)
        # Joint-policy closure without a physical flag is a task truncation.
        return bool(terminated and not truncated), bool(truncated or not terminated)

    def _output(self, snapshot, result=None):
        n, p = len(self.agent_ids), len(self.physical_agent_ids)
        if tuple(snapshot.global_state.agent_ids) != self.physical_agent_ids:
            raise ValueError("Global-state order must match the configured physical agents")
        active = np.array([aid in snapshot.agents for aid in self.agent_ids])[:, None]
        masks = snapshot.global_state.masks
        physical = {}
        for key in ("active", "controlled", "trainable", "finished", "crashed",
                    "truncated", "task_complete"):
            fallback = [aid in snapshot.active_physical_agents if key == "active"
                        else aid in self.agent_ids if key == "trainable"
                        else key == "controlled" for aid in self.physical_agent_ids]
            physical[key] = self._tensor(
                np.asarray(masks.get(f"{key}_mask", fallback)).reshape(p, 1), torch.bool)
        root_term, root_trunc = self._physical_boundary(result)
        learning_done = not bool(snapshot.agents)
        output = TensorDict({
            "state": self._tensor(snapshot.global_state.vector, torch.float32),
            "agents": TensorDict({
                "observation": self._tensor(self._observations, torch.float32),
                "index": torch.arange(n, dtype=torch.int64, device=self.device),
                "active": self._tensor(active, torch.bool),
                "terminated": self._tensor(self._terminated, torch.bool),
                "truncated": self._tensor(self._truncated, torch.bool),
                "done": self._tensor(self._terminated | self._truncated, torch.bool),
            }, batch_size=(n,), device=self.device),
            "physical": TensorDict(physical, batch_size=(p,), device=self.device),
            "learning": TensorDict({
                "done": self._tensor([learning_done], torch.bool),
                "terminated": self._tensor([learning_done], torch.bool),
                "truncated": self._tensor([False], torch.bool),
                "valid": self._tensor([bool(result and result.decisions)], torch.bool),
            }, batch_size=(), device=self.device),
            "done": self._tensor([snapshot.episode_done], torch.bool),
            "terminated": self._tensor([root_term], torch.bool),
            "truncated": self._tensor([root_trunc], torch.bool),
            "race_episode_done": self._tensor([snapshot.episode_done], torch.bool),
            "decision_steps": self._tensor([snapshot.decision_steps], torch.int64),
            "physics_steps": self._tensor([snapshot.physics_steps], torch.int64),
            "decision_physics_steps": self._tensor([result.physics_steps if result else 0], torch.int64),
            "agent_steps": self._tensor([result.agent_steps if result else 0], torch.int64),
            "elapsed_seconds": self._tensor([result.elapsed_seconds if result else 0], torch.float32),
        }, batch_size=(), device=self.device)
        if result is not None:
            rewards = self.reward_mapping.from_step(result, self.agent_ids)
            output["agents", "reward"] = self._tensor(
                [[rewards.get(aid, 0.0)] for aid in self.agent_ids], torch.float32)
            output["agents", "individual_reward"] = self._tensor(
                [[result.decisions[aid].individual_reward if aid in result.decisions else 0.0]
                 for aid in self.agent_ids], torch.float32)
            output["team_reward"] = self._tensor(
                [next(iter(rewards.values())) if rewards and self.reward_mapping.mode == "team_shared" else 0.0],
                torch.float32)
            output["shared_bonus"] = self._tensor([result.team_reward], torch.float32)
        return output

    def render(self):
        return self.task.render() if self.render_mode is not None else None

    def close(self, *, raise_if_closed=True):
        if not self.is_closed:
            self.task.close()
            super().close(raise_if_closed=raise_if_closed)
