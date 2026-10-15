"""Native TorchRL translation of RaceTask with fixed learner slots."""
from __future__ import annotations

import numpy as np
import torch
from tensordict import NonTensorData, TensorDict, TensorDictBase
from torchrl.data import Bounded, Categorical, Composite, NonTensor, Unbounded
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
    _NUMPY_DTYPES = {torch.float32: np.float32, torch.int64: np.int64, torch.bool: np.bool_}

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
                 team_reward_reduction="mean", device="cpu", environment_id=None,
                 batch_reports=False):
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
        self.profile_call = None
        self._pending_seed = None
        self.environment_id = environment_id
        self.batch_reports = batch_reports
        self.environment_episode = -1
        self.report_events = []
        n, p = len(self.agent_ids), len(self.physical_agent_ids)
        self._observations = np.zeros((n, max(width[0] for width in widths)), dtype=np.float32)
        self._terminated = np.zeros((n, 1), dtype=bool)
        self._truncated = np.zeros((n, 1), dtype=bool)
        self.register_buffer("_agent_index", torch.arange(n, dtype=torch.int64, device=self.device))

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
        if environment_id is not None:
            self.observation_spec["environment_id"] = count((1,))
            self.observation_spec["environment_episode"] = count((1,))
        if environment_id is not None and not batch_reports:
            # Direct consumers retain the per-transition reporting contract;
            # process collectors attach buffered reports once per worker batch.
            self.observation_spec["race_events"] = NonTensor(shape=(), example_data=[])
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
        # Own a NumPy copy before wrapping it: task/scratch arrays can change on
        # the next step, whereas retained TensorDict transitions must not.
        tensor = torch.from_numpy(np.array(value, dtype=self._NUMPY_DTYPES[dtype], copy=True))
        return tensor if self.device.type == "cpu" else tensor.to(self.device)

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
        self.environment_episode += 1
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
        def policy_actions():
            if not active:
                return {}
            detached = action.detach()
            cpu = (self.profile_call("action_device_to_cpu", detached.cpu, device=action.device)
                   if self.profile_call and action.device.type != "cpu" else detached.cpu())
            actions = cpu.numpy().copy()
            return {aid: actions[i] for i, aid in enumerate(self.agent_ids) if aid in active}
        actions = (self.profile_call("action_conversion", policy_actions)
                   if self.profile_call else policy_actions())
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
        if self.environment_id is not None and not self.batch_reports:
            output["race_events"] = NonTensorData(self.report_events)
            self.report_events = []
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
        masks = snapshot.global_state.masks
        physical_keys = ("active", "controlled", "trainable", "finished", "crashed",
                         "truncated", "task_complete")
        # Fresh, owned packs are filled directly from task/scratch storage.
        # Only disjoint slices of these per-output allocations are shared;
        # no reusable working buffer is exposed to a collector.
        boolean_values = np.empty(7 * p + 4 * n + 8, dtype=np.bool_)
        physical_values = boolean_values[:7 * p].reshape(7, p, 1)
        for i, key in enumerate(physical_keys):
            value = masks.get(f"{key}_mask")
            if value is None:
                value = [aid in snapshot.active_physical_agents if key == "active"
                         else aid in self.agent_ids if key == "trainable"
                         else key == "controlled" for aid in self.physical_agent_ids]
            physical_values[i] = np.asarray(value).reshape(p, 1)
        agent_values = boolean_values[7 * p:7 * p + 4 * n].reshape(4, n, 1)
        agent_values[0, :, 0] = [aid in snapshot.agents for aid in self.agent_ids]
        agent_values[1] = self._terminated
        agent_values[2] = self._truncated
        np.logical_or(self._terminated, self._truncated, out=agent_values[3])
        root_term, root_trunc = self._physical_boundary(result)
        learning_done = not bool(snapshot.agents)
        boolean_values[-8:] = (snapshot.episode_done, root_term, root_trunc, snapshot.episode_done,
                               learning_done, learning_done, False, bool(result and result.decisions))
        if self.device.type == "cpu":
            # NumPy views avoid Torch slice/reshape dispatch on the CPU path.
            physical_tensor = torch.from_numpy(physical_values)
            agent_tensor = torch.from_numpy(agent_values)
            root_tensor = torch.from_numpy(boolean_values[-8:].reshape(8, 1))
        else:
            boolean_tensor = torch.from_numpy(boolean_values).to(self.device)
            physical_pack, agent_pack, root_pack = boolean_tensor.split((7 * p, 4 * n, 8))
            physical_tensor = physical_pack.view(7, p, 1)
            agent_tensor = agent_pack.view(4, n, 1)
            root_tensor = root_pack.view(8, 1)
        physical = dict(zip(physical_keys, physical_tensor.unbind(0)))
        agent_active, agent_term, agent_trunc, agent_done = agent_tensor.unbind(0)
        done, terminated, truncated, race_done, learning_end, learning_term, learning_trunc, valid = root_tensor.unbind(0)

        counts = [snapshot.decision_steps, snapshot.physics_steps,
                  result.physics_steps if result else 0, result.agent_steps if result else 0]
        if self.environment_id is not None:
            counts.extend((self.environment_id, self.environment_episode))
        count_tensors = self._tensor(counts, torch.int64).view(-1, 1).unbind(0)
        decision_steps, physics_steps, decision_physics_steps, agent_steps = count_tensors[:4]

        state = snapshot.global_state.vector
        state_end = state.size
        obs_end = state_end + self._observations.size
        real_values = np.empty(obs_end + 1 + (2 * n + 2 if result is not None else 0), dtype=np.float32)
        real_values[:state_end] = state.reshape(-1)
        real_values[state_end:obs_end] = self._observations.reshape(-1)
        real_values[obs_end] = result.elapsed_seconds if result else 0
        if result is not None:
            rewards = (self.profile_call("race_bookkeeping_reward", self.reward_mapping.from_step,
                                         result, self.agent_ids)
                       if self.profile_call else self.reward_mapping.from_step(result, self.agent_ids))
            real_values[obs_end + 1:obs_end + 1 + n] = [rewards.get(aid, 0.0) for aid in self.agent_ids]
            real_values[obs_end + 1 + n:obs_end + 1 + 2 * n] = [
                result.decisions[aid].individual_reward if aid in result.decisions else 0.0
                for aid in self.agent_ids]
            real_values[-2:] = (next(iter(rewards.values()))
                                if rewards and self.reward_mapping.mode == "team_shared" else 0.0,
                                result.team_reward)
        if self.device.type == "cpu":
            state_tensor = torch.from_numpy(real_values[:state_end].reshape(state.shape))
            observation_tensor = torch.from_numpy(real_values[state_end:obs_end].reshape(self._observations.shape))
            elapsed_tensor = torch.from_numpy(real_values[obs_end:obs_end + 1])
            if result is not None:
                reward_tensor = torch.from_numpy(real_values[obs_end + 1:-2].reshape(2, n, 1))
                bonus_tensor = torch.from_numpy(real_values[-2:].reshape(2, 1))
        else:
            real_tensor = torch.from_numpy(real_values).to(self.device)
            state_tensor = real_tensor[:state_end].view(state.shape)
            observation_tensor = real_tensor[state_end:obs_end].view(self._observations.shape)
            elapsed_tensor = real_tensor[obs_end:obs_end + 1]
            if result is not None:
                reward_tensor = real_tensor[obs_end + 1:-2].view(2, n, 1)
                bonus_tensor = real_tensor[-2:].view(2, 1)
        agents = {
            "observation": observation_tensor,
            "index": self._agent_index.clone(),
            "active": agent_active,
            "terminated": agent_term,
            "truncated": agent_trunc,
            "done": agent_done,
        }
        if result is not None:
            agents["reward"], agents["individual_reward"] = reward_tensor.unbind(0)
        values = {
            "state": state_tensor,
            "agents": TensorDict(agents, batch_size=(n,), device=self.device),
            "physical": TensorDict(physical, batch_size=(p,), device=self.device),
            "learning": TensorDict({
                "done": learning_end,
                "terminated": learning_term,
                "truncated": learning_trunc,
                "valid": valid,
            }, batch_size=(), device=self.device),
            "done": done,
            "terminated": terminated,
            "truncated": truncated,
            "race_episode_done": race_done,
            "decision_steps": decision_steps,
            "physics_steps": physics_steps,
            "decision_physics_steps": decision_physics_steps,
            "agent_steps": agent_steps,
            "elapsed_seconds": elapsed_tensor,
        }
        if result is not None:
            values["team_reward"], values["shared_bonus"] = bonus_tensor.unbind(0)
        if self.environment_id is not None:
            values["environment_id"], values["environment_episode"] = count_tensors[4:]
        if self.environment_id is not None and not self.batch_reports:
            values["race_events"] = NonTensorData([])
        return TensorDict(values, batch_size=(), device=self.device)

    def render(self):
        return self.task.render() if self.render_mode is not None else None

    def close(self, *, raise_if_closed=True):
        if not self.is_closed:
            resources = getattr(self, "profile_resources", None)
            if resources is not None:
                resources.close()
            self.task.close()
            super().close(raise_if_closed=raise_if_closed)
