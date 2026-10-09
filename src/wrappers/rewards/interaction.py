"""Teammate progress and local blocking rewards."""
from __future__ import annotations

import numpy as np

from wrappers.rewards.base import RewardComponent


class TeamSupportComponent(RewardComponent):
    """Credit teammate progress and local, non-contact blocking outcomes.

    Blocking credit is the teammate's signed progress advantage over an active
    opponent following behind the ego in the same corridor. There is no reward
    for proximity alone, parked/crashed opponents, reversing, or a collision.
    This is an outcome proxy, not a claim that ego caused the opponent's delay.
    """

    def __init__(self, config):
        self.teammate_id = str(config.get('teammate_id', 'car_0'))
        self.progress_weight = float(config.get('progress_weight', 1.0))
        self.blocking_weight = float(config.get('blocking_weight', 1.0))
        self.max_distance = float(config.get('max_distance', 6.0))
        self.min_distance = float(config.get('min_distance', 0.6))
        self.lateral_distance = float(config.get('lateral_distance', 0.6))
        self.max_step_progress = float(config.get('max_step_progress', 1.0))
        values = [self.progress_weight, self.blocking_weight, self.max_distance,
                  self.min_distance, self.lateral_distance, self.max_step_progress]
        if (not all(np.isfinite(v) and v >= 0 for v in values)
                or not self.max_distance > self.min_distance
                or self.lateral_distance <= 0 or self.max_step_progress <= 0):
            raise ValueError('team_support requires finite nonnegative weights and positive distance/progress bounds')

    def compute(self, step_info):
        infos = step_info.get('all_infos') or {}
        ego = step_info.get('info') or {}
        mate = infos.get(self.teammate_id, {})
        # No bonus for a mutual crash or for continuing after the racer is gone.
        if (ego.get('terminal_reason') or mate.get('terminal_reason')
                or (ego.get('track_limits') or {}).get('exceeded')
                or (mate.get('track_limits') or {}).get('exceeded')):
            return {}
        length = step_info.get('track_length')
        if not length or not np.isfinite(length) or length <= 0:
            raise ValueError('team_support requires a positive track_length')
        if not mate.get('centerline') or not ego.get('centerline'):
            raise ValueError('team_support requires ego and teammate centerline facts')
        def delta(info):
            return float(np.clip(info['centerline'].get('progress_delta', 0.0) * length,
                                 -self.max_step_progress, self.max_step_progress))
        mate_delta = delta(mate)
        rewards = {'team_support/progress': self.progress_weight * mate_delta}
        opponents = set(step_info.get('opponent_agent_ids') or ())
        advantages = []
        if delta(ego) >= 0 and mate_delta > 0:
            for neighbor in ego.get('frenet_neighbors', []):
                aid = neighbor['agent_id']
                other = infos.get(aid, {})
                if (aid not in opponents or not other.get('centerline')
                        or other.get('terminal_reason')
                        or (other.get('track_limits') or {}).get('exceeded')):
                    continue
                if (-self.max_distance <= neighbor['delta_s'] <= -self.min_distance
                        and abs(neighbor['delta_d']) <= self.lateral_distance):
                    # Negative opponent motion cannot manufacture blocking credit.
                    advantages.append(mate_delta - max(0.0, delta(other)))
        if advantages:
            rewards['team_support/blocking'] = self.blocking_weight * sum(advantages) / max(1, len(opponents))
        return rewards
