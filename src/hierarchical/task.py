"""Single-agent categorical macro decisions around the unchanged RaceTask."""
from dataclasses import replace
import json
import time
import numpy as np

from env.spaces import SpaceSpec
from tasks.contracts import TaskStep
from hierarchical.selectors import SkillExecutor, make_selector, geometry_annotation


class HierarchicalTask:
    hierarchical = True

    def __init__(self, task, bank, config):
        if len(task.possible_agents) != 1 or task.fixed_policy_agents:
            raise ValueError('Initial hierarchical task requires one solo racer')
        self.task, self.bank, self.config = task, bank, config
        self.aid = task.possible_agents[0]
        self.executor = SkillExecutor(bank, config.get('interval', 5))
        self.gamma = float(config.get('gamma', .99))
        if not 0 < self.gamma <= 1:
            raise ValueError('LL gamma must be in (0, 1]')
        self.n_skills = len(bank.actors)
        self.selector = make_selector(config.get('selector', {}), bank.manifest)
        self.dim = bank.obs_dim + self.n_skills + 1 + bank.action_dim
        self._snapshot = None
        self.last_duration = 0
        self.total_ll = self.total_hl = self.total_seconds = self.episodes = 0
        self.selection_counts = np.zeros(self.n_skills, dtype=np.int64)
        self.execution_counts = np.zeros(self.n_skills, dtype=np.int64)
        self.transitions = np.zeros((self.n_skills, self.n_skills), dtype=np.int64)
        self.annotations = None
        self.ll_inference_seconds = 0.

    def __getattr__(self, name):
        if name == 'task':
            raise AttributeError(name)
        return getattr(self.task, name)

    @property
    def spec(self):
        original = self.task.spec
        contract = {'version': 1, 'low_level': original.observation_contracts[self.aid],
                    'extensions': ['active_skill_one_hot', 'll_decisions_since_switch', 'previous_action'],
                    'reset': {'active_skill': 'all_zero', 'since_switch': 0, 'previous_action': 0}}
        return replace(original, observation_dims={self.aid: self.dim},
            observation_contracts={self.aid: contract},
            action_lows={self.aid: np.array([0], dtype=np.int64)},
            action_highs={self.aid: np.array([self.n_skills - 1], dtype=np.int64)})

    def observation_space(self, agent):
        return SpaceSpec((self.dim,), -np.inf, np.inf)

    def action_space(self, agent):
        return SpaceSpec((1,), 0, self.n_skills - 1, dtype=np.int64)

    def observation(self, ll):
        state = self.executor.state
        one_hot = np.zeros(self.n_skills, dtype=np.float32)
        if state.active >= 0:
            one_hot[state.active] = 1
        return np.concatenate((ll, one_hot, [state.since_change], self.executor.last_action)).astype(np.float32)

    def snapshot(self, snapshot):
        return replace(snapshot, observations={aid: self.observation(obs) for aid, obs in snapshot.observations.items()})

    def reset(self, *, seed=None, options=None):
        snapshot = self.task.reset(seed=seed, options=options)
        self.executor.reset()
        self.last_duration = 0
        self.trace = []
        self.previous_physical_action = None
        self.dwell_times = []
        self.skill_distance = np.zeros(self.n_skills)
        self.episode_ll = self.episode_hl = 0
        self.selection_counts.fill(0)
        self.execution_counts.fill(0)
        self.transitions.fill(0)
        self.elapsed = self.distance = 0.
        self.episodes += 1
        self.selector = make_selector(self.config.get('selector', {}), self.bank.manifest, seed or 0)
        self.annotations = None
        if self.selector is not None or self.config.get('record_trajectory', False):
            path = self.task.env.yaml_path.with_name(self.task.env.yaml_path.stem + '_metadata.json')
            if path.exists():
                self.annotations = json.loads(path.read_text())
        self._snapshot = snapshot
        return self.snapshot(snapshot)

    def step(self, actions, *, on_physics_step=None):
        if self._snapshot is None or self.episode_done:
            raise RuntimeError('Reset before hierarchical stepping')
        if set(actions) != {self.aid}:
            raise ValueError('Supply exactly the active solo skill action')
        action = np.asarray(actions[self.aid])
        if action.shape != (1,) or action.dtype.kind not in 'iu' or not 0 <= action[0] < self.n_skills:
            raise ValueError('HL action requires one valid integer skill ID')
        skill = int(action[0])
        if self.selector is not None:
            skill = self.selector.select(metadata=self.annotations,
                s=self._snapshot.infos[self.aid].get('centerline', {}).get('s', 0.0))
        before = self.snapshot(self._snapshot)
        old = self.executor.state.active
        old_dwell = self.executor.state.since_change
        self.executor.state.select(skill)
        self.selection_counts[skill] += 1
        if old >= 0 and old != skill:
            self.dwell_times.append(old_dwell)
        if old >= 0:
            self.transitions[old, skill] += 1
        reward, components, substeps = 0., {}, []
        for i in range(self.executor.interval):
            previous_action = self.executor.last_action.copy()
            inference_started = time.perf_counter()
            ll_action = self.executor.action(self._snapshot.observations[self.aid])
            self.ll_inference_seconds += time.perf_counter() - inference_started
            result = self.task.step({self.aid: ll_action}, on_physics_step=on_physics_step)
            decision = result.decisions[self.aid]
            physical = np.asarray(decision.action_physical)
            physical_previous = physical if self.previous_physical_action is None else self.previous_physical_action
            self.elapsed += result.elapsed_seconds
            self.episode_ll += 1
            distance = sum(max(0., substep.facts.info[self.aid].get('centerline', {}).get('progress_delta', 0.))
                           for substep in result.substeps) * self.task.env.centerline_track_length
            self.distance += distance
            self.skill_distance[skill] += distance
            if self.config.get('record_trajectory', False):
                state = result.substeps[-1].facts.agent_states[self.aid]
                self.trace.append({**geometry_annotation(self.annotations, decision.info.get('centerline', {}).get('s', 0.), self.bank.manifest[skill]['name']),
                    'timestamp': self.elapsed, 'decision_dt': result.elapsed_seconds, 'distance_m': self.distance, 'x': float(state.pose[0]), 'y': float(state.pose[1]),
                    'skill': skill, 'action': ll_action.tolist(),
                    'physical_action': physical.tolist(),
                    'physical_steering_command_jump': float(abs(physical[0] - physical_previous[0])),
                    'physical_speed_command_jump': float(abs(physical[1] - physical_previous[1])),
                    'command_jump': float(np.linalg.norm(ll_action - previous_action)),
                    'steering_jump': float(abs(ll_action[0] - previous_action[0])),
                    'speed_command_jump': float(abs(ll_action[1] - previous_action[1])),
                    'collision': bool(decision.info.get('collision', False)),
                    'terminated': bool(decision.terminated), 'truncated': bool(decision.truncated),
                    's': float(decision.info.get('centerline', {}).get('s', 0.)),
                    'lateral_error': float(decision.info.get('centerline', {}).get('d', 0.)),
                    'speed': float(np.linalg.norm(state.velocity)),
                    'switched': old >= 0 and old != skill and i == 0})
            self.previous_physical_action = physical.copy()
            reward += self.gamma ** i * decision.individual_reward
            for name, value in decision.reward_components.items():
                components[name] = components.get(name, 0.) + self.gamma ** i * value
            substeps.extend(result.substeps)
            self._snapshot = result.after
            if decision.terminated or decision.truncated or result.after.episode_done:
                break
        k = i + 1
        self.last_duration = k
        self.total_ll += k
        self.total_hl += 1
        self.episode_hl += 1
        self.execution_counts[skill] += k
        self.total_seconds += sum(s.timestep for s in substeps)
        final = replace(decision, observation=before.observations[self.aid],
            action_normalized=np.array([skill], dtype=np.int64), individual_reward=reward,
            reward_components=components, next_observation=self.observation(decision.next_observation),
            info={**decision.info, 'hierarchical': {'skill': skill, 'duration': k,
                'discount': self.gamma ** k, 'switches': self.executor.state.switches}})
        return TaskStep(before, self.snapshot(result.after), {self.aid: final}, 0., {}, tuple(substeps), ll_decisions=k)

    def metrics(self):
        state = self.executor.state
        metrics = {'hierarchical/ll_decisions': self.episode_ll,
            'hierarchical/hl_decisions': self.episode_hl,
            'hierarchical/switches': state.switches,
            'hierarchical/switches_per_metre': state.switches / max(self.distance, 1e-8),
            'hierarchical/switches_per_second': state.switches / max(self.elapsed, 1e-8),
            'hierarchical/mean_dwell_ll_decisions': float(np.mean(self.dwell_times + [state.since_change])),
            'hierarchical/ll_inference_seconds_total': self.ll_inference_seconds,
            'hierarchical/bank_initialization_seconds': self.bank.initialization_seconds}
        for entry in self.bank.manifest:
            i, name = entry['id'], entry['name']
            metrics[f'hierarchical/selection_fraction/{name}'] = float(self.selection_counts[i] / max(self.episode_hl, 1))
            metrics[f'hierarchical/execution_ll_decisions/{name}'] = int(self.execution_counts[i])
            metrics[f'hierarchical/distance_m/{name}'] = float(self.skill_distance[i])
        return metrics
