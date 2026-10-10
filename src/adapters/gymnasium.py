from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.error import ResetNeeded

from adapters.common import TaskAdapter, box
from tasks import RaceTaskProtocol


class RaceGymEnv(TaskAdapter, gym.Env):
    def __init__(self, task: RaceTaskProtocol) -> None:
        if len(task.possible_agents) != 1:
            raise ValueError("The Gymnasium adapter requires exactly one policy agent")
        if getattr(task, "team_reward_agent_id", None) is not None:
            raise ValueError("The Gymnasium adapter exposes individual rewards; shared bonuses require a team adapter")
        super().__init__(task)
        self.agent_id = task.possible_agents[0]
        self.observation_space = box(task.observation_space(self.agent_id))
        self.action_space = box(task.action_space(self.agent_id))
        self._needs_reset = True

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        snapshot = self._reset_task(seed, options)
        self._needs_reset = False
        return snapshot.observations[self.agent_id].copy(), self._info(self.agent_id)

    def step(self, action):
        if self._needs_reset:
            raise ResetNeeded("Reset the adapter before stepping a new episode")
        result = self._step_task({self.agent_id: np.asarray(action, dtype=np.float32)})
        decision = result.decisions[self.agent_id]
        self._needs_reset = decision.terminated or decision.truncated
        return (
            decision.next_observation.copy(), decision.individual_reward,
            decision.terminated, decision.truncated, self._info(self.agent_id),
        )
