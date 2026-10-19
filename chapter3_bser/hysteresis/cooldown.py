from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ReplanningCooldown:
    cooldown_steps: int = 20
    last_replan_step: int | None = None

    def remaining(self, step: int) -> int:
        if self.last_replan_step is None:
            return 0
        return max(0, int(self.cooldown_steps) - (int(step) - int(self.last_replan_step)))

    def ready(self, step: int) -> bool:
        return self.remaining(step) == 0

    def mark(self, step: int) -> None:
        self.last_replan_step = int(step)
