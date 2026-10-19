from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class TriggerDecision:
    should_replan: bool
    reason: str
    objective_gain: float
    cooldown_remaining: int
