"""Single-agent macro decisions around RaceTask."""
from dataclasses import replace
import json
import time
import numpy as np

from env.spaces import SpaceSpec
from tasks.contracts import TaskStep
from hierarchical.selectors import SkillExecutor, CompositionExecutor, make_selector, geometry_annotation
from hierarchical.composition import CompositionExecution, coefficient_transform, composition_spec


class HierarchicalTask:
    hierarchical = True

    def __init__(self, task, bank, config):
        if len(task.possible_agents) != 1 or task.fixed_policy_agents:
            raise ValueError('Initial hierarchical task requires one solo racer')
        self.task, self.bank, self.config = task, bank, config
        self.aid = task.possible_agents[0]
        self.composition = config.get('composition', {})
        self.distribution = self.composition.get('distribution', 'categorical')
        self.action_contract = composition_spec(self.composition)
        self.categorical = self.action_contract.categorical
        self.temperature = float(self.composition.get('temperature', 1.))
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError('Composition temperature must be finite and positive')
        self.influence_every = self.composition.get('influence_every', 0)
        if type(self.influence_every) is not int or self.influence_every < 0:
            raise ValueError('Composition influence_every must be a nonnegative macro decision count')
        self.executor = (SkillExecutor if self.categorical else CompositionExecutor)(bank, config.get('interval', 5))
        self.gamma = float(config.get('gamma', .99))
        if not 0 < self.gamma <= 1:
            raise ValueError('LL gamma must be in (0, 1]')
        scale = config.get('dwell_time_scale_seconds', 1.0)
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not np.isfinite(scale) or scale <= 0:
            raise ValueError('dwell_time_scale_seconds must be finite and positive')
        self.dwell_scale = float(scale)
        self.preview = dict(config.get('preview', {}))
        if type(self.preview.get('enabled', False)) is not bool:
            raise ValueError('HL preview enabled must be boolean')
        if self.preview.get('enabled', False):
            count = self.preview.get('samples', 20)
            if type(count) is not int or count < 1:
                raise ValueError('HL preview samples must be a positive integer')
            for key, default in (('spacing_m', .6), ('curvature_scale', 1.), ('width_scale', 1.)):
                value = self.preview.get(key, default)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
                    raise ValueError(f'HL preview {key} must be finite and positive')
                self.preview[key] = float(value)
            self.preview['samples'] = count
        self.n_skills = len(bank.manifest)
        self.action_dim = self.action_contract.action_dim
        self.selector = make_selector(config.get('selector', {}), bank.manifest)
        if not self.categorical and self.selector is not None:
            raise ValueError('Composition requires the learned selector')
        self.dim = bank.obs_dim + self.n_skills + 1 + bank.action_dim + (2 * self.preview['samples'] if self.preview.get('enabled', False) else 0)
        self.selector_rng = np.random.default_rng(np.random.SeedSequence([int(task.env.seed), 0x48524c]))
        self.dwell_seconds = 0.
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
        contract = {'version': 2, 'low_level': original.observation_contracts[self.aid],
                    'extensions': ['active_skill_one_hot', 'tanh_dwell_seconds', 'previous_action'],
                    'dwell_time_scale_seconds': self.dwell_scale,
                    'preview': self.preview,
                    'reset': {'active_skill': 'all_zero', 'since_switch': 0, 'previous_action': 0}}
        if not self.categorical:
            contract['extensions'][0] = 'previous_composition_coefficients'
            contract['composition_distribution'] = self.distribution
            contract['residual_layers'] = self.bank.layer_contract
            contract['reset'] = {'composition': 'all_zero', 'since_dominant_change': 0, 'previous_action': 0}
        space = self.action_space(self.aid)
        return replace(original, observation_dims={self.aid: self.dim},
            observation_contracts={self.aid: contract},
            action_lows={self.aid: space.low}, action_highs={self.aid: space.high})

    def observation_space(self, agent):
        return SpaceSpec((self.dim,), -np.inf, np.inf)

    def action_space(self, agent):
        if self.categorical:
            return SpaceSpec((1,), 0, self.n_skills - 1, dtype=np.int64)
        return SpaceSpec((self.action_dim,), 0 if self.action_contract.simplex else -np.inf,
                         1 if self.action_contract.simplex else np.inf)

    def observation(self, ll):
        state = self.executor.state
        one_hot = np.zeros(self.n_skills, dtype=np.float32)
        if not self.categorical:
            one_hot = self.executor.coefficients
        elif state.active >= 0:
            one_hot[state.active] = 1
        parts = [ll, one_hot, [np.tanh(self.dwell_seconds / self.dwell_scale)], self.executor.last_action]
        if self.preview.get('enabled', False):
            geometry = self.task.env._track_preview_geometry
            if geometry is None:
                raise ValueError('HL preview requires existing track preview geometry')
            position = self.task.env.get_agent_state(self.aid).pose[:2]
            preview = geometry.preview(position, self.preview['samples'], spacing=self.preview['spacing_m'])
            parts.extend((preview['curvature'] / self.preview['curvature_scale'],
                          preview['width'] / self.preview['width_scale']))
        return np.concatenate(parts).astype(np.float32)

    def snapshot(self, snapshot):
        return replace(snapshot, observations={aid: self.observation(obs) for aid, obs in snapshot.observations.items()})

    def reset(self, *, seed=None, options=None):
        snapshot = self.task.reset(seed=seed, options=options)
        self.executor.reset()
        self.last_duration = 0
        self.trace = []
        self.composition_count = 0
        self.composition_sum = np.zeros(self.n_skills)
        self.composition_abs_sum = np.zeros(self.n_skills)
        self.composition_outer = np.zeros((self.n_skills, self.n_skills))
        self.composition_min = np.full(self.n_skills, np.inf)
        self.composition_max = np.full(self.n_skills, -np.inf)
        self.composition_entropy = self.residual_entropy = self.composition_change = 0.
        self.composition_effective = self.negative_count = self.zero_residual = 0.
        self.influence_sums, self.influence_count = {}, 0
        self.previous_physical_action = None
        self.fallback_counts = {}
        self.dwell_times = []
        self.dwell_durations = []
        self.dwell_seconds = 0.
        self.episode_ll_return = self.episode_macro_return = self.episode_discounted_ll_return = 0.
        self.episode_inference_seconds = 0.
        self.net_progress_m = self.physical_distance_m = 0.
        self.last_position = np.asarray(self.task.env.get_agent_state(self.aid).pose[:2], dtype=float).copy()
        self.skill_distance = np.zeros(self.n_skills)
        self.episode_ll = self.episode_hl = 0
        self.selection_counts.fill(0)
        self.execution_counts.fill(0)
        self.transitions.fill(0)
        self.elapsed = self.distance = 0.
        self.episodes += 1
        if seed is not None:
            self.selector_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0x48524c]))
        selector_seed = int(self.selector_rng.integers(0, 2**63))
        self.selector = make_selector(self.config.get('selector', {}), self.bank.manifest, selector_seed)
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
        selected = actions[self.aid]
        evaluation_coefficients = selected.coefficients if isinstance(selected, CompositionExecution) else None
        action = np.asarray(evaluation_coefficients if evaluation_coefficients is not None else selected)
        if self.categorical:
            if action.shape != (1,) or action.dtype.kind not in 'iu' or not 0 <= action[0] < self.n_skills:
                raise ValueError('HL action requires one valid integer skill ID')
            skill = int(action[0])
        else:
            if evaluation_coefficients is not None:
                if action.shape != (self.n_skills,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
                    raise ValueError('Invalid evaluation composition coefficients')
                if self.action_contract.convex:
                    if np.any(action < 0) or not np.isclose(action.sum(), 1., atol=1e-5):
                        raise ValueError('Convex evaluation coefficients must lie on the simplex')
                elif action[0] != 0 or (self.action_contract.coefficient_constraint == 'independent_positive'
                                        and np.any(action < 0)):
                    raise ValueError('Invalid independent residual coefficients')
                coefficients = action.astype(np.float32, copy=True)
                # The evaluation runner does not optimize this synthetic action.
                action = coefficients if self.action_contract.simplex else coefficients[1:]
            elif action.shape != (self.action_dim,) or not np.isfinite(action).all():
                raise ValueError('Invalid composition action')
            elif self.action_contract.simplex:
                if np.any(action < 0) or not np.isclose(action.sum(), 1., atol=1e-5):
                    raise ValueError('Dirichlet action must lie on the simplex')
                coefficients = action.astype(np.float32, copy=True)
            else:
                coefficients = coefficient_transform(
                    action, self.action_contract, self.temperature).astype(np.float32, copy=False)
                if not self.action_contract.convex:
                    coefficients = np.concatenate((np.zeros(1, dtype=np.float32), coefficients))
            skill = int(np.argmax(np.abs(coefficients)))
            self._record_composition(coefficients)
        if self.selector is not None:
            skill = self.selector.select(metadata=self.annotations,
                s=self._snapshot.infos[self.aid].get('centerline', {}).get('s', 0.0))
            reason = getattr(self.selector, 'last_fallback_reason', None)
            if reason:
                self.fallback_counts[reason] = self.fallback_counts.get(reason, 0) + 1
        before = self.snapshot(self._snapshot)
        old = self.executor.state.active
        old_dwell = self.executor.state.since_change
        self.executor.state.select(skill)
        if not self.categorical:
            self.executor.coefficients = coefficients
        if old != skill:
            if old >= 0:
                self.dwell_durations.append(self.dwell_seconds)
            self.dwell_seconds = 0.
        self.selection_counts[skill] += 1
        if old >= 0 and old != skill:
            self.dwell_times.append(old_dwell)
        if old >= 0:
            self.transitions[old, skill] += 1
        reward, components, substeps = 0., {}, []
        for i in range(self.executor.interval):
            previous_action = self.executor.last_action.copy()
            inference_started = time.perf_counter()
            if not self.categorical and self.influence_every and i == 0 and self.episode_hl % self.influence_every == 0:
                ll_action = self.executor.action(self._snapshot.observations[self.aid], collect_influence=True)
                self.influence_count += 1
                for name, value in self.executor.last_influence.items():
                    self.influence_sums[name] = self.influence_sums.get(name, 0.) + value
            else:
                ll_action = self.executor.action(self._snapshot.observations[self.aid])
            inference_elapsed = time.perf_counter() - inference_started
            self.ll_inference_seconds += inference_elapsed
            self.episode_inference_seconds += inference_elapsed
            result = self.task.step({self.aid: ll_action}, on_physics_step=on_physics_step)
            decision = result.decisions[self.aid]
            physical = np.asarray(decision.action_physical)
            physical_previous = physical if self.previous_physical_action is None else self.previous_physical_action
            self.elapsed += result.elapsed_seconds
            self.dwell_seconds += result.elapsed_seconds
            ll_reward = decision.individual_reward
            self.episode_ll_return += ll_reward
            self.episode_discounted_ll_return += self.gamma ** self.episode_ll * ll_reward
            self.episode_ll += 1
            distance = sum(max(0., substep.facts.info[self.aid].get('centerline', {}).get('progress_delta', 0.))
                           for substep in result.substeps) * self.task.env.centerline_track_length
            self.net_progress_m += sum(substep.facts.info[self.aid].get('centerline', {}).get('progress_delta', 0.)
                                       for substep in result.substeps) * self.task.env.centerline_track_length
            before_distance = self.distance
            for substep in result.substeps:
                position = np.asarray(substep.facts.agent_states[self.aid].pose[:2], dtype=float)
                self.physical_distance_m += float(np.linalg.norm(position - self.last_position))
                self.last_position = position.copy()
            self.distance += distance
            self.skill_distance[skill] += distance
            if self.config.get('record_trajectory', False):
                state = result.substeps[-1].facts.agent_states[self.aid]
                self.trace.append({**geometry_annotation(self.annotations, decision.info.get('centerline', {}).get('s', 0.), self.bank.manifest[skill]['name']),
                    'timestamp': self.elapsed, 'decision_dt': result.elapsed_seconds, 'distance_m': self.distance, 'forward_centerline_progress_m': self.distance, 'before_forward_centerline_progress_m': before_distance, 'x': float(state.pose[0]), 'y': float(state.pose[1]),
                    **({'skill': skill} if self.categorical else {'dominant_option': skill}), 'action': ll_action.tolist(),
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
                    **({'switched': old >= 0 and old != skill and i == 0} if self.categorical else
                       {'dominant_changed': old >= 0 and old != skill and i == 0})})
            self.previous_physical_action = physical.copy()
            reward += self.gamma ** i * decision.individual_reward
            for name, value in decision.reward_components.items():
                components[name] = components.get(name, 0.) + self.gamma ** i * value
            substeps.extend(result.substeps)
            self._snapshot = result.after
            if decision.terminated or decision.truncated or result.after.episode_done:
                break
        k = i + 1
        self.episode_macro_return += reward
        self.last_duration = k
        self.total_ll += k
        self.total_hl += 1
        self.episode_hl += 1
        self.execution_counts[skill] += k
        self.total_seconds += sum(s.timestep for s in substeps)
        transition_info = ({'skill': skill, 'switches': self.executor.state.switches} if self.categorical else
                           {'dominant_option': skill, 'dominant_transitions': self.executor.state.switches})
        final = replace(decision, observation=before.observations[self.aid],
            action_normalized=np.array([skill], dtype=np.int64) if self.categorical else action.copy(), individual_reward=reward,
            reward_components=components, next_observation=self.observation(decision.next_observation),
            info={**decision.info, 'hierarchical': {**transition_info, 'duration': k, 'discount': self.gamma ** k}})
        return TaskStep(before, self.snapshot(result.after), {self.aid: final}, 0., {}, tuple(substeps), ll_decisions=k)

    def _record_composition(self, coefficients):
        c = np.asarray(coefficients, dtype=np.float64)
        if self.composition_count:
            self.composition_change += float(np.abs(c - self.executor.coefficients).sum())
        self.composition_count += 1
        self.composition_sum += c
        self.composition_abs_sum += np.abs(c)
        self.composition_outer += np.outer(c, c)
        self.composition_min = np.minimum(self.composition_min, c)
        self.composition_max = np.maximum(self.composition_max, c)
        self.negative_count += float((c[1:] < 0).sum())
        def entropy(values):
            values = np.abs(values)
            if not values.sum():
                return 0.
            p = values[values > 0] / values.sum()
            return float(-np.sum(p * np.log(p)))
        residual = entropy(c[1:])
        self.residual_entropy += residual
        self.composition_entropy += entropy(c)
        if np.abs(c[1:]).sum() == 0:
            self.zero_residual += 1
        else:
            self.composition_effective += np.exp(residual)

    def metrics(self):
        state = self.executor.state
        probabilities = self.selection_counts / max(self.episode_hl, 1)
        positive = probabilities[probabilities > 0]
        metrics = {'hierarchical/ll_decisions': self.episode_ll,
            'hierarchical/hl_decisions': self.episode_hl,
            'hierarchical/selection_entropy': float(-np.sum(positive * np.log(positive))),
            'hierarchical/switches': state.switches,
            'hierarchical/switches_per_metre': state.switches / max(self.distance, 1e-8),
            'hierarchical/switches_per_second': state.switches / max(self.elapsed, 1e-8),
            'hierarchical/mean_dwell_ll_decisions': float(np.mean(self.dwell_times + [state.since_change])),
            'hierarchical/ll_inference_seconds_total': self.ll_inference_seconds,
            'hierarchical/ll_inference_seconds_episode': self.episode_inference_seconds,
            'hierarchical/return_ll_undiscounted': self.episode_ll_return,
            'hierarchical/return_macro_discounted_sum': self.episode_macro_return,
            'hierarchical/return_ll_globally_discounted': self.episode_discounted_ll_return,
            'hierarchical/net_centerline_progress_m': self.net_progress_m,
            'hierarchical/forward_centerline_progress_m': self.distance,
            'hierarchical/physical_distance_m': self.physical_distance_m,
            'hierarchical/elapsed_simulated_seconds': self.elapsed,
            'hierarchical/mean_dwell_seconds': float(np.mean(self.dwell_durations + [self.dwell_seconds])),
            'hierarchical/bank_initialization_seconds': self.bank.initialization_seconds}
        if hasattr(self.selector, 'last_fallback_reason'):
            metrics['hierarchical/geometry_fallback_fraction'] = sum(self.fallback_counts.values()) / max(self.episode_hl, 1)
            for reason, count in self.fallback_counts.items():
                metrics[f'hierarchical/geometry_fallback_count/{reason}'] = count
        metrics['hierarchical/bank_tensor_storage_bytes'] = self.bank.accounting()['tensor_bytes']
        for entry in self.bank.manifest:
            i, name = entry['id'], entry['name']
            metrics[f'hierarchical/selection_count/{name}'] = int(self.selection_counts[i])
            metrics[f'hierarchical/selection_fraction/{name}'] = float(self.selection_counts[i] / max(self.episode_hl, 1))
            metrics[f'hierarchical/execution_ll_decisions/{name}'] = int(self.execution_counts[i])
            metrics[f'hierarchical/forward_centerline_progress_m/{name}'] = float(self.skill_distance[i])
            # Legacy alias: this has always measured positive centerline progress.
            metrics[f'hierarchical/distance_m/{name}'] = float(self.skill_distance[i])
        if not self.categorical and self.composition_count:
            for key in ('selection_entropy', 'switches', 'switches_per_metre', 'switches_per_second',
                        'mean_dwell_ll_decisions', 'mean_dwell_seconds'):
                metrics['composition/dominant_' + key.replace('switches', 'transitions').replace('selection_', '')] = metrics.pop('hierarchical/' + key)
            for entry in self.bank.manifest:
                name = entry['name']
                for key in ('selection_count', 'selection_fraction', 'execution_ll_decisions', 'forward_centerline_progress_m', 'distance_m'):
                    metrics[f'composition/dominant_{key}/{name}'] = metrics.pop(f'hierarchical/{key}/{name}')
            n = self.composition_count
            mean = self.composition_sum / n
            covariance = self.composition_outer / n - np.outer(mean, mean)
            metrics.update({'composition/residual_entropy': self.residual_entropy / n,
                'composition/routing_entropy': self.composition_entropy / n,
                'composition/effective_residual_adapters': self.composition_effective / max(n - self.zero_residual, 1),
                'composition/zero_residual_fraction': self.zero_residual / n,
                'composition/negative_fraction': self.negative_count / (n * (self.n_skills - 1)),
                'composition/total_absolute_residual_weight': float(self.composition_abs_sum[1:].sum() / n),
                'composition/mean_l1_change': self.composition_change / max(n - 1, 1),
                'composition/dominant_transitions': state.switches})
            metrics['composition/influence_samples'] = self.influence_count
            for name, value in self.influence_sums.items():
                metrics[f'composition/influence/{name}'] = value / self.influence_count
            for i, entry in enumerate(self.bank.manifest):
                name = entry['name']
                metrics.update({f'composition/mean/{name}': float(mean[i]),
                    f'composition/std/{name}': float(np.sqrt(max(covariance[i, i], 0.))),
                    f'composition/min/{name}': float(self.composition_min[i]),
                    f'composition/max/{name}': float(self.composition_max[i]),
                    f'composition/mean_absolute/{name}': float(self.composition_abs_sum[i] / n)})
                for j, other in enumerate(self.bank.manifest):
                    metrics[f'composition/covariance/{name}/{other["name"]}'] = float(covariance[i, j])
        return metrics
