"""Validate learner objectives against task reward capabilities."""
from typing import Dict, Sequence


def validate_team_reward_composers(composers: Dict, *, trainable_ids: Sequence[str],
                                   opponent_ids: Sequence[str], reward_mode: str,
                                   critic_mode: str, team_return_mode: str,
                                   action_repeat: int) -> bool:
    """Reject shared reward terms unless collection can credit the whole team."""
    contracts = [getattr(composers.get(aid), "team_contract", []) for aid in trainable_ids]
    if not any(contracts):
        return False
    if any(contract != contracts[0] for contract in contracts):
        raise ValueError("All teammates must use identical team reward components")
    if (len(trainable_ids) != 2 or len(opponent_ids) != 2
            or reward_mode != "team_shared" or critic_mode != "shared_team"
            or team_return_mode != "joint" or action_repeat != 1):
        raise ValueError("Team race rewards require 2v2 MAPPO with joint team returns, shared_team critic, team_shared reward, and action_repeat=1")
    return True
