from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from typing import Any, Callable, Optional, Sequence

import numpy as np

from env.spaces import SpaceSpec
from tasks.contracts import AgentDecision, TaskSnapshot, TaskStep, TaskSubstep
from tasks.reward_context import build_reward_context


def _detach(value, memo=None):
    if memo is None:
        memo = {}
    if id(value) in memo:
        return memo[id(value)]
    if isinstance(value, np.ndarray):
        result = value.copy()
        if not value.flags.writeable:
            result.setflags(write=False)
    elif isinstance(value, Mapping):
        result = {}
        memo[id(value)] = result
        result.update({key: _detach(item, memo) for key, item in value.items()})
    elif isinstance(value, (list, tuple, set, frozenset)):
        result = type(value)(_detach(item, memo) for item in value)
    elif is_dataclass(value) and not isinstance(value, type):
        result = replace(value, **{field.name: _detach(getattr(value, field.name), memo)
                                  for field in fields(value) if field.init})
    else:
        return value
    memo[id(value)] = result
    return result


class RaceTask:
    def __init__(
        self, env: Any, *, policy_agents: Sequence[str], fixed_controllers: Mapping,
        obs_composers: Mapping, reward_composers: Mapping, action_composers: Mapping,
        action_repeat: int = 1, team_reward_agent_id: Optional[str] = None,
    ) -> None:
        self.env = env
        self.physical_agents = tuple(env.possible_agents)
        self.possible_agents = tuple(policy_agents)
        self.fixed_policy_agents = tuple(fixed_controllers)
        policy_set, fixed_set = set(self.possible_agents), set(self.fixed_policy_agents)
        if (len(policy_set) != len(self.possible_agents) or policy_set & fixed_set
                or policy_set | fixed_set != set(self.physical_agents)):
            raise ValueError("Each physical agent must have exactly one action owner")
        for composers in (obs_composers, reward_composers, action_composers):
            if set(composers) != policy_set:
                raise ValueError("Composer keys must match policy agents")
            if len({id(composer) for composer in composers.values()}) != len(composers):
                raise ValueError("Each policy agent needs independent composers")
        if (isinstance(action_repeat, bool) or not isinstance(action_repeat, int)
                or action_repeat < 1):
            raise ValueError("action_repeat must be a positive integer")
        team_contracts = [getattr(composer, "team_contract", [])
                          for composer in reward_composers.values()]
        if any(team_contracts):
            if any(contract != team_contracts[0] for contract in team_contracts):
                raise ValueError("All teammates must use identical team reward components")
            if team_reward_agent_id is None:
                team_reward_agent_id = self.possible_agents[0]
        if team_reward_agent_id is not None and team_reward_agent_id not in policy_set:
            raise ValueError("Team reward composer must belong to a policy agent")
        self.action_repeat = action_repeat
        self.fixed_controllers = dict(fixed_controllers)
        self.obs_composers = dict(obs_composers)
        self.reward_composers = dict(reward_composers)
        self.action_composers = dict(action_composers)
        self.team_reward_agent_id = team_reward_agent_id
        self._snapshot: Optional[TaskSnapshot] = None
        self._decision_steps = 0
        self._physics_steps = 0
        self._closed = False

    @property
    def agents(self):
        return self._snapshot.agents if self._snapshot is not None else ()

    @property
    def episode_done(self):
        return self._snapshot.episode_done if self._snapshot is not None else False

    @property
    def timestep(self):
        return float(self.env.timestep)

    @property
    def render_mode(self):
        return self.env.render_mode

    def action_space(self, agent: str) -> SpaceSpec:
        if agent not in self.possible_agents:
            raise KeyError(agent)
        return SpaceSpec((2,), -1.0, 1.0)

    def observation_space(self, agent: str) -> SpaceSpec:
        composer = self.obs_composers[agent]
        return SpaceSpec((composer.obs_dim,), -np.inf, np.inf)

    def state_space(self) -> SpaceSpec:
        return SpaceSpec(self.env.get_global_state().vector.shape, -np.inf, np.inf)

    def _make_snapshot(self, raw_obs, infos, observations, global_state):
        active = tuple(self.env.agents)
        agents = tuple(aid for aid in self.possible_agents if aid in active)
        return TaskSnapshot(
            agents=agents, active_physical_agents=active,
            observations={aid: observations[aid].copy() for aid in agents},
            raw_observations=_detach(raw_obs), infos=_detach(infos),
            global_state=_detach(global_state), decision_steps=self._decision_steps,
            physics_steps=self._physics_steps,
            episode_done=bool(self.env.episode_done) or not active,
        )

    def reset(self, *, seed=None, options=None) -> TaskSnapshot:
        raw_obs, infos = self.env.reset(seed=seed, options=options)
        for controller in self.fixed_controllers.values():
            reset = getattr(controller, "reset", None)
            if reset is not None:
                reset()
        for composers in (self.action_composers, self.obs_composers, self.reward_composers):
            for composer in composers.values():
                reset = getattr(composer, "reset", None)
                if reset is not None:
                    reset()
        self._decision_steps = self._physics_steps = 0
        observations = {aid: self.obs_composers[aid].wrap(raw_obs.get(aid, {}), infos.get(aid, {}))
                        for aid in self.possible_agents if aid in self.env.agents}
        self._snapshot = self._make_snapshot(raw_obs, infos, observations, self.env.get_global_state())
        return self._snapshot

    def step(self, actions: Mapping[str, np.ndarray], *,
             on_physics_step: Optional[Callable[[TaskSubstep], None]] = None) -> TaskStep:
        if self._snapshot is None or self.episode_done:
            raise RuntimeError("Reset the task before stepping a new episode")
        before = self._snapshot
        if set(actions) != set(before.agents):
            raise ValueError("Action keys must exactly match active policy agents")
        normalized = {}
        for aid in before.agents:
            action = np.asarray(actions[aid])
            if (action.dtype != np.float32 or action.shape != (2,)
                    or not np.isfinite(action).all() or (np.abs(action) > 1).any()):
                raise ValueError(f"{aid}: expected a finite float32 (2,) action in [-1, 1]")
            normalized[aid] = action.copy()
        physical = {aid: np.asarray(self.action_composers[aid].process(action.copy()),
                                    dtype=np.float32).copy()
                    for aid, action in normalized.items()}
        for aid, controller in self.fixed_controllers.items():
            if aid in before.active_physical_agents:
                try:
                    action = controller.act(before.raw_observations[aid])
                except Exception:
                    action = np.zeros(2, dtype=np.float32)
                physical[aid] = np.asarray(action, dtype=np.float32).copy()

        rewards = dict.fromkeys(before.agents, 0.0)
        components = {aid: {} for aid in before.agents}
        terminated = dict.fromkeys(before.agents, False)
        truncated = dict.fromkeys(before.agents, False)
        team_reward, team_components, substeps = 0.0, {}, []
        for _ in range(self.action_repeat):
            supplied_actions = _detach(physical)
            raw_obs, _, terms, truncs, infos = self.env.step(physical)
            self._physics_steps += 1
            facts = _detach(self.env.last_step_facts)
            global_state = facts.global_state
            substep = TaskSubstep(self._physics_steps, self.timestep, supplied_actions, facts)
            substeps.append(substep)
            if on_physics_step is not None:
                on_physics_step(substep)
            for aid in before.agents:
                agent_term, agent_trunc = bool(terms.get(aid, False)), bool(truncs.get(aid, False))
                terminated[aid] |= agent_term
                truncated[aid] |= agent_trunc
                context = build_reward_context(
                    env=self.env, agent_id=aid, info_dict=infos, obs_dict=raw_obs,
                    actions=physical, global_state=global_state,
                )
                context.update(obs=before.observations[aid], next_obs=raw_obs.get(aid, {}),
                               info=infos.get(aid, {}), done=agent_term or agent_trunc,
                               terminated=agent_term, truncated=agent_trunc,
                               action=normalized[aid], timestep=self.timestep)
                reward, breakdown = self.reward_composers[aid].compute(context)
                rewards[aid] += float(reward)
                for name, value in breakdown.items():
                    components[aid][name] = components[aid].get(name, 0.0) + float(value)
            if self.team_reward_agent_id is not None:
                aid = self.team_reward_agent_id
                context = build_reward_context(
                    env=self.env, agent_id=aid, info_dict=infos, obs_dict=raw_obs,
                    actions=physical, global_state=global_state,
                )
                bonus, breakdown = self.reward_composers[aid].compute(context, team=True)
                team_reward += float(bonus)
                for name, value in breakdown.items():
                    team_components[name] = team_components.get(name, 0.0) + float(value)
            if self.env.episode_done or not set(physical).issubset(self.env.agents):
                break

        next_observations, decisions = {}, {}
        for aid in before.agents:
            info = _detach(infos.get(aid, {}))
            if aid not in self.env.agents and not (terminated[aid] or truncated[aid]):
                truncated[aid] = True
                info["task_boundary"] = "episode_policy"
            next_obs = self.obs_composers[aid].wrap(raw_obs.get(aid, {}), info)
            next_observations[aid] = np.asarray(next_obs, dtype=np.float32).copy()
            decisions[aid] = AgentDecision(
                before.observations[aid].copy(), normalized[aid], physical[aid].copy(),
                rewards[aid], components[aid], next_observations[aid],
                terminated[aid], truncated[aid], info,
            )
        self._decision_steps += 1
        self._snapshot = self._make_snapshot(raw_obs, infos, next_observations, global_state)
        return TaskStep(before, self._snapshot, decisions, team_reward, team_components, tuple(substeps))

    def render(self):
        return self.env.render()

    def close(self) -> None:
        if not self._closed:
            self.env.close()
            self._closed = True
