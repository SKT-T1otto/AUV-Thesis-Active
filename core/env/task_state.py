from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from typing import Tuple
Vector3 = Tuple[float, float, float]

@dataclass(frozen=True)
class MissionTaskState:
    step: int
    target_found: bool
    finder_id: int
    executor_knows_target: bool
    handoff_step: Optional[int]
    mission_complete: bool
    completion_step: Optional[int]

@dataclass(frozen=True)
class AgentStateView:
    role_order: Tuple[str, ...]
    positions: Tuple[Vector3, ...]
    velocities: Tuple[Vector3, ...]
    navigation_targets: Tuple[Vector3, ...]
    collision_flags: Tuple[bool, ...]

@dataclass(frozen=True)
class SearchExecutionState:
    searcher_ids: Tuple[int, int, int]
    executor_id: int
    target_known_by_agent: Tuple[bool, ...]
    waypoint_reached_counts: Tuple[int, ...]
    agent_finished: Tuple[bool, ...]
    hold_counters: Tuple[int, ...]
