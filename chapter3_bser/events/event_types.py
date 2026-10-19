from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

class BSEREvent(str, Enum):
    BELIEF_SHIFT = 'BELIEF_SHIFT'
    OBSTACLE_DISCOVERED = 'OBSTACLE_DISCOVERED'
    TARGET_FOUND = 'TARGET_FOUND'
    EXECUTOR_TARGET_RECEIVED = 'EXECUTOR_TARGET_RECEIVED'
    EXECUTOR_PUBLIC_TARGET_UPDATED = 'EXECUTOR_PUBLIC_TARGET_UPDATED'
    TARGET_LOST = 'TARGET_LOST'
    EXECUTOR_INVALID = 'EXECUTOR_INVALID'
    WAYPOINT_STALE = 'WAYPOINT_STALE'
    PERIODIC_REFRESH = 'PERIODIC_REFRESH'

@dataclass(frozen=True)
class EventDetection:
    events: Tuple[BSEREvent, ...]
    previous_entropy: float
    current_entropy: float
    belief_distance: float
    belief_shift_score: float
    new_obstacle_cells: int
    new_obstacle_probability_mass: float
    risk_change: float
    executor_reachable: bool
    stale_searcher_ids: Tuple[int, ...]
    executor_target_received: bool = False
    executor_planning_cost_relative_change: float = 0.0
    assignment_waypoints: Tuple[Tuple[int, Tuple[float, float, float]], ...] = ()
    executor_validity_evaluated: bool = True
    executor_validity_deferred: bool = False
    executor_invalid_reason: str = ''
    executor_query_failure_reason: str = ''
    executor_assignment_reachable: bool = True
    executor_query_reachable: bool = True
    executor_start_endpoint_current: bool = True
    executor_goal_endpoint_current: bool = True
    executor_installed_planning_cost: float | None = None
    executor_current_planning_cost: float | None = None
    executor_public_target_shift: float = 0.0
