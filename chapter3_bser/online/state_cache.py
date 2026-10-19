from __future__ import annotations
from dataclasses import dataclass
from core.mapping.planning_state import PlanningStateView
from core.mapping.planning_state import planning_state_sha256

@dataclass
class OnlineStateCache:
    previous: PlanningStateView | None = None
    current: PlanningStateView | None = None
    previous_sha256: str | None = None
    current_sha256: str | None = None

    def initialize(self, state: PlanningStateView) -> None:
        digest = planning_state_sha256(state)
        self.previous = state
        self.current = state
        self.previous_sha256 = digest
        self.current_sha256 = digest

    def update(self, state: PlanningStateView) -> None:
        if self.current is None:
            self.initialize(state)
            return
        self.previous = self.current
        self.previous_sha256 = self.current_sha256
        self.current = state
        self.current_sha256 = planning_state_sha256(state)
