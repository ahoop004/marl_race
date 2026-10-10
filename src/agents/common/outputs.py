"""Policy inference results are explicit and independent of rollout storage."""
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PolicyOutput:
    actions: Any
    log_probs: Any
    raw_actions: Any
    values: Any = None
