from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
Vector3 = tuple[float, float, float]

@dataclass
class _TrackingState:
    signature: tuple[tuple[Vector3, ...], Vector3]
    next_index: int = 0

@dataclass(frozen=True)
class PathTrackerSnapshot:
    """Immutable, read-only view of one agent's local path-tracking state."""
    current_target: Vector3 | None
    next_index: int
    path_point_count: int
    remaining_point_count: int
    completed: bool
    final_waypoint: Vector3 | None
    remaining_path_points: tuple[Vector3, ...] = ()

class PathTracker:
    """Convert a high-level path into one ordered local tracking target."""

    def __init__(self, threshold: float=0.75):
        if float(threshold) <= 0.0:
            raise ValueError('path tracking threshold must be positive')
        self.threshold = float(threshold)
        self._state: dict[int, _TrackingState] = {}
        self._last_targets: dict[int, Vector3] = {}

    @staticmethod
    def _vector(value: Iterable[float]) -> Vector3:
        vector = tuple((float(item) for item in value))
        if len(vector) != 3:
            raise ValueError('path tracking targets must be 3-D')
        return vector

    def reset(self, agent_id: int | None=None) -> None:
        if agent_id is None:
            self._state.clear()
            self._last_targets.clear()
            return
        self._state.pop(int(agent_id), None)
        self._last_targets.pop(int(agent_id), None)

    def prune(self, active_agent_ids: Iterable[int]) -> None:
        active = {int(value) for value in active_agent_ids}
        for agent_id in tuple(self._state):
            if agent_id not in active:
                self.reset(agent_id)

    def tracking_target(self, agent_id: int, current_position: Iterable[float], planned_path: Iterable[Iterable[float]], final_waypoint: Iterable[float]) -> Vector3:
        agent_id = int(agent_id)
        path = tuple((self._vector(point) for point in planned_path))
        final = self._vector(final_waypoint)
        signature = (path, final)
        tracking = self._state.get(agent_id)
        if tracking is None or tracking.signature != signature:
            tracking = _TrackingState(signature=signature)
            self._state[agent_id] = tracking
        if tracking.next_index < len(path):
            current_target = path[tracking.next_index]
            distance = float(np.linalg.norm(np.asarray(current_position, dtype=np.float64) - np.asarray(current_target, dtype=np.float64)))
            if distance < self.threshold:
                tracking.next_index += 1
        target = path[tracking.next_index] if tracking.next_index < len(path) else final
        self._last_targets[agent_id] = target
        return target

    def snapshot(self, agent_id: int) -> PathTrackerSnapshot:
        """Return diagnostics without advancing the path index or target."""
        agent_id = int(agent_id)
        tracking = self._state.get(agent_id)
        if tracking is None:
            return PathTrackerSnapshot(None, 0, 0, 0, True, None, ())
        (path, final) = tracking.signature
        next_index = int(tracking.next_index)
        remaining = tuple(path[next_index:])
        if not remaining or remaining[-1] != final:
            remaining = (*remaining, final)
        return PathTrackerSnapshot(current_target=self._last_targets.get(agent_id), next_index=next_index, path_point_count=len(path), remaining_point_count=max(0, len(path) - next_index), completed=next_index >= len(path), final_waypoint=final, remaining_path_points=remaining)
