"""Baseline selectors. Geometry annotations are privileged diagnostics only."""
from dataclasses import dataclass
import numpy as np


@dataclass
class SkillState:
    active: int = -1
    previous: int = -1
    since_change: int = 0
    since_selection: int = 0
    switches: int = 0

    def select(self, skill):
        self.previous = self.active
        if skill != self.active:
            self.switches += int(self.active >= 0)
            self.since_change = 0
        self.active = int(skill)
        self.since_selection = 0

    def advance(self):
        self.since_change += 1
        self.since_selection += 1


class FixedSelector:
    def __init__(self, skill=0):
        self.skill = int(skill)

    def select(self, **context):
        return self.skill


class RandomSelector:
    def __init__(self, n_skills, seed=0):
        self.n_skills = n_skills
        self.rng = np.random.default_rng(seed)

    def select(self, **context):
        return int(self.rng.integers(self.n_skills))


class SequenceSelector:
    """Reproducible switch-pair approaches for offline compatibility studies."""
    def __init__(self, sequence):
        self.sequence = tuple(sequence)
        self.position = 0

    def select(self, **context):
        # Execute the approach once, then retain the final skill for measuring stability.
        skill = self.sequence[min(self.position, len(self.sequence) - 1)]
        self.position += 1
        return skill


class GeometrySelector:
    """Phase5 V2 arc-length annotations; unsupported corners/support/transition use fallback.

    Look ahead along the periodic centerline by anticipation metres. This is a
    rule-based comparison, never an optimal oracle or a learned observation.
    """
    def __init__(self, skills, anticipation=1.0, fallback=0):
        self.ids = {entry['name']: entry['id'] for entry in skills}
        self.anticipation = float(anticipation)
        self.fallback = int(fallback)
        self.last_fallback_reason = None
        if self.anticipation < 0:
            raise ValueError('anticipation must be nonnegative')

    def select(self, *, metadata=None, s=0.0, **context):
        self.last_fallback_reason = None
        if not metadata or metadata.get('schema_version') != '2.0.0' or not metadata.get('measured', {}).get('length_m'):
            self.last_fallback_reason = 'missing_metadata'
            return self.fallback
        length = metadata['measured']['length_m']
        query = (s + self.anticipation) % length
        matches = []
        for maneuver in metadata.get('maneuvers', []):
            start, end = maneuver['s_start_m'], maneuver['s_end_m']
            inside = (query >= start or query < end) if maneuver['wraps_seam'] else start <= query < end
            if inside:
                matches.append(maneuver)
        if len(matches) != 1:
            self.last_fallback_reason = 'ambiguous_annotation' if matches else 'no_matching_maneuver'
        else:
            maneuver = matches[0]
            family = maneuver['type']
            role = maneuver.get('requested', {}).get('composition_role')
            if role != 'primary':
                self.last_fallback_reason = 'transition_region' if role in {'transition', 'distributed_gap'} else 'supporting_maneuver'
            elif family not in self.ids:
                self.last_fallback_reason = 'unsupported_geometry_family'
            else:
                return self.ids[family]
        return self.fallback



def make_selector(config, skills, seed=0):
    kind = config.get('type', 'learned')
    if kind == 'learned':
        return None
    if kind == 'fixed':
        selector = FixedSelector(config.get('skill', 0))
        if not 0 <= selector.skill < len(skills):
            raise ValueError('Invalid fixed skill')
        return selector
    if kind == 'random':
        return RandomSelector(len(skills), seed)
    if kind == 'sequence':
        sequence = config.get('sequence', [])
        if not sequence or any(type(skill) is not int or not 0 <= skill < len(skills) for skill in sequence):
            raise ValueError('Sequence selector requires a nonempty list of valid skill IDs')
        return SequenceSelector(sequence)
    if kind == 'geometry':
        fallback = config.get('fallback', 0)
        if type(fallback) is not int or not 0 <= fallback < len(skills):
            raise ValueError('Invalid geometry fallback skill')
        return GeometrySelector(skills, config.get('anticipation', 1.0), fallback)
    raise ValueError(f'Unknown hierarchical selector: {kind}')


class SkillExecutor:
    """One instance per environment; bank may be shared by many executors."""
    def __init__(self, bank, interval=5):
        if type(interval) is not int or interval < 1:
            raise ValueError('selection interval must be a positive LL decision count')
        self.bank, self.interval = bank, interval
        self.reset()

    def reset(self):
        self.state = SkillState()
        self.last_action = np.zeros(self.bank.action_dim, dtype=np.float32)

    def action(self, observation, skill=None):
        import torch
        if skill is not None:
            if isinstance(skill, bool) or not isinstance(skill, (int, np.integer)) or not 0 <= skill < len(self.bank.actors):
                raise ValueError('Unknown skill ID')
            self.state.select(skill)
        if self.state.active < 0:
            raise RuntimeError('Select a skill before driving')
        self.last_action = self.bank(np.asarray(observation)[None],
            torch.tensor([self.state.active], dtype=torch.long))[0].cpu().numpy()
        self.state.advance()
        return self.last_action.copy()


def geometry_annotation(metadata, s, skill_name=None):
    """Diagnostic labels only; shared by offline trajectory reports."""
    if not metadata:
        return {'geometry': 'unknown', 'maneuver_id': 'unknown', 'composition_role': 'unknown'}
    length = metadata['measured']['length_m']
    s %= length
    relevant = [m for m in metadata.get('maneuvers', []) if m['type'] == skill_name
                and m.get('requested', {}).get('composition_role') == 'primary']
    entry_distance = min(((m['s_start_m'] - s) % length for m in relevant), default=None)
    for m in metadata.get('maneuvers', []):
        inside = ((s >= m['s_start_m'] or s < m['s_end_m']) if m['wraps_seam']
                  else m['s_start_m'] <= s < m['s_end_m'])
        if inside:
            return {'geometry': m['type'], 'maneuver_id': m['id'],
                'composition_role': m.get('requested', {}).get('composition_role', 'unknown'),
                'distance_since_entry_m': (s - m['s_start_m']) % length,
                'selection_distance_to_skill_entry_m': entry_distance}
    return {'geometry': 'unknown', 'maneuver_id': 'unknown', 'composition_role': 'unknown'}
