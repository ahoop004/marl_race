from __future__ import annotations

import numpy as np
from gymnasium import spaces
from gymnasium.error import ResetNeeded

from env.spaces import SpaceSpec
from tasks import RaceTaskProtocol, TaskSnapshot, TaskStep
from tasks.race_task import _detach


def box(spec: SpaceSpec) -> spaces.Box:
    return spaces.Box(low=spec.low.copy(), high=spec.high.copy(), dtype=spec.dtype)


class TaskAdapter:
    @classmethod
    def from_scenario(cls, scenario, *, scenario_dir=None, mode="train", render_mode=None):
        from core.task_builder import create_race_task

        task = create_race_task(scenario, scenario_dir=scenario_dir, mode=mode, render_mode=render_mode)
        try:
            return cls(task)
        except BaseException:
            task.close()
            raise

    def __init__(self, task: RaceTaskProtocol) -> None:
        self.task = task
        self.render_mode = task.render_mode
        if self.render_mode not in (None, "human", "rgb_array"):
            raise ValueError(f"Unsupported render mode: {self.render_mode!r}")
        self.metadata = {
            "name": "marl_race_v0", "render_modes": ["human", "rgb_array"],
            "render_fps": max(1, round(1 / (task.timestep * task.action_repeat))),
        }
        self.state_space = box(task.state_space())
        self.snapshot: TaskSnapshot | None = None
        self.last_step: TaskStep | None = None

    def _reset_task(self, seed=None, options=None) -> TaskSnapshot:
        snapshot = self.task.reset(seed=seed, options=options)
        if set(snapshot.agents) != set(self.task.possible_agents):
            raise ValueError("Reset must activate every exposed policy agent")
        self.snapshot, self.last_step = snapshot, None
        if self.render_mode == "human":
            self.task.render()
        return snapshot

    def _on_physics_step(self, substep) -> None:
        if self.render_mode == "human":
            self.task.render()

    def _step_task(self, actions) -> TaskStep:
        result = self.task.step(actions, on_physics_step=self._on_physics_step)
        self.snapshot, self.last_step = result.after, result
        return result

    def _info(self, agent: str) -> dict:
        if self.snapshot is None:
            raise ResetNeeded("Reset the adapter before reading its state")
        snapshot = self.snapshot
        result = self.last_step
        decision = result.decisions.get(agent) if result is not None else None
        info = _detach(decision.info if decision is not None else snapshot.infos.get(agent, {}))
        info.update(
            global_state=snapshot.global_state.vector.copy(),
            global_state_agent_ids=tuple(snapshot.global_state.agent_ids),
            global_state_version=snapshot.global_state.metadata.get("vector_contract_version"),
            lifecycle_masks=_detach(snapshot.global_state.masks),
            active_physical_agents=snapshot.active_physical_agents,
            race_episode_done=snapshot.episode_done,
            decision_steps=snapshot.decision_steps, physics_steps=snapshot.physics_steps,
            individual_reward=0.0, reward_components={},
            team_reward=0.0, team_reward_components={},
            decision_physics_steps=0, agent_steps=0, elapsed_seconds=0.0,
        )
        if result is not None:
            if decision is not None:
                info.update(individual_reward=decision.individual_reward,
                            reward_components=dict(decision.reward_components))
            info.update(
                team_reward=result.team_reward,
                team_reward_components=dict(result.team_reward_components),
                decision_physics_steps=result.physics_steps,
                agent_steps=result.agent_steps, elapsed_seconds=result.elapsed_seconds,
            )
        return info

    def state(self) -> np.ndarray:
        if self.snapshot is None:
            raise ResetNeeded("Reset the adapter before reading its state")
        return self.snapshot.global_state.vector.copy()

    def advance_fixed_agents(self) -> TaskStep:
        if self.snapshot is None:
            raise ResetNeeded("Reset the adapter before advancing the race")
        if self.task.agents:
            raise RuntimeError("Policy agents still need actions")
        return self._step_task({})

    def render(self):
        if self.render_mode is None:
            return None
        if self.snapshot is None:
            raise ResetNeeded("Reset the adapter before rendering")
        return self.task.render()

    def close(self) -> None:
        self.task.close()
