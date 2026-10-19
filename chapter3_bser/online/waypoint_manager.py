from __future__ import annotations
from dataclasses import replace
import numpy as np
from chapter3_bser.online.types import CurrentAssignment
from chapter3_bser.online.types import OnlineAllocation

class WaypointManager:

    def __init__(self, tolerance: float=1e-09, *, agent_cooldown_steps: int=0):
        self.tolerance = float(tolerance)
        self.agent_cooldown_steps = int(agent_cooldown_steps)
        self.last_switch_step: dict[int, int] = {}
        self.switch_count = 0
        self.total_switch_distance = 0.0
        self.maximum_switch_distance = 0.0
        self.same_agent_switch_within_10_steps = 0
        self._current_assignment: OnlineAllocation | None = None

    @property
    def current_assignment(self) -> OnlineAllocation | None:
        return self._current_assignment

    def commit(self, allocation: OnlineAllocation | None) -> None:
        self._current_assignment = allocation

    def stabilize(self, previous: OnlineAllocation, proposed: OnlineAllocation, *, affected_agent_ids, step: int, route_valid_by_agent=None) -> OnlineAllocation:
        affected = {int(value) for value in affected_agent_ids}
        valid = dict(route_valid_by_agent or {})
        old = {item.agent_id: item for item in previous.search_assignments}
        selected = []
        for item in proposed.search_assignments:
            prior = old.get(item.agent_id)
            if prior is None:
                selected.append(item)
                continue
            distance = float(np.linalg.norm(np.asarray(item.waypoint) - np.asarray(prior.waypoint)))
            within_cooldown = item.agent_id in self.last_switch_step and int(step) - self.last_switch_step[item.agent_id] < self.agent_cooldown_steps
            preserve = bool(item.agent_id not in affected or item.candidate_id == prior.candidate_id or distance < self.tolerance or (within_cooldown and valid.get(item.agent_id, True)))
            selected.append(prior if preserve else item)
        return replace(proposed, search_assignments=tuple(sorted(selected, key=lambda value: value.agent_id)))

    def updates(self, previous: OnlineAllocation | None, current: OnlineAllocation, *, reason: str, step: int) -> tuple[CurrentAssignment, ...]:
        old = {} if previous is None else {item.agent_id: item.waypoint for item in previous.search_assignments}
        output = []
        for item in current.search_assignments:
            old_waypoint = old.get(item.agent_id)
            changed = old_waypoint is None or float(np.linalg.norm(np.asarray(item.waypoint) - np.asarray(old_waypoint))) > self.tolerance
            if changed:
                distance = 0.0 if old_waypoint is None else float(np.linalg.norm(np.asarray(item.waypoint) - np.asarray(old_waypoint)))
                previous_step = self.last_switch_step.get(item.agent_id)
                if previous_step is not None and int(step) - previous_step <= 10:
                    self.same_agent_switch_within_10_steps += 1
                self.last_switch_step[item.agent_id] = int(step)
                self.switch_count += 1
                self.total_switch_distance += distance
                self.maximum_switch_distance = max(self.maximum_switch_distance, distance)
                output.append(CurrentAssignment(item.agent_id, old_waypoint, item.waypoint, str(reason), int(step)))
        return tuple(sorted(output, key=lambda item: item.agent_id))
