from __future__ import annotations

import numpy as np
from gymnasium.error import ResetNeeded
from pettingzoo import ParallelEnv

from adapters.common import TaskAdapter, box
from tasks import RaceTaskProtocol


class RaceParallelEnv(TaskAdapter, ParallelEnv):
    def __init__(self, task: RaceTaskProtocol) -> None:
        if not task.possible_agents:
            raise ValueError("The PettingZoo adapter requires policy agents")
        super().__init__(task)
        self.possible_agents = list(task.possible_agents)
        self.agents = []
        self.observation_spaces = {aid: box(task.observation_space(aid)) for aid in self.possible_agents}
        self.action_spaces = {aid: box(task.action_space(aid)) for aid in self.possible_agents}

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        snapshot = self._reset_task(seed, options)
        self.agents = list(snapshot.agents)
        observations = {aid: snapshot.observations[aid].copy() for aid in self.agents}
        return observations, {aid: self._info(aid) for aid in self.agents}

    def step(self, actions):
        if self.snapshot is None:
            raise ResetNeeded("Reset the adapter before stepping a new episode")
        if not self.agents:
            if actions:
                raise ValueError("There are no active policy agents")
            return {}, {}, {}, {}, {}
        result = self._step_task({aid: np.asarray(action, dtype=np.float32)
                                  for aid, action in actions.items()})
        self.agents = list(result.after.agents)
        observations = {aid: decision.next_observation.copy() for aid, decision in result.decisions.items()}
        rewards = {aid: decision.individual_reward for aid, decision in result.decisions.items()}
        terminated = {aid: decision.terminated for aid, decision in result.decisions.items()}
        truncated = {aid: decision.truncated for aid, decision in result.decisions.items()}
        return observations, rewards, terminated, truncated, {aid: self._info(aid) for aid in result.decisions}
