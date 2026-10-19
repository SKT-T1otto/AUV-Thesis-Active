from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from typing import Optional
from typing import Tuple
from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.events.event_types import EventDetection
Vector3 = Tuple[float, float, float]

@dataclass(frozen=True)
class SearchAssignment:
    agent_id: int
    candidate_id: str
    waypoint: Vector3
    path: Tuple[Vector3, ...]
    physical_travel_time: float
    path_cell_indices: Tuple[int, ...] = ()
    planning_cost: float = float('inf')
    failure_reason: Optional[str] = None

@dataclass(frozen=True)
class ExecutorAssignment:
    executor_id: int
    target_region: Vector3
    path: Tuple[Vector3, ...]
    estimated_arrival_time: float
    source: str
    reachable: bool
    path_cell_indices: Tuple[int, ...] = ()
    planning_cost: float = float('inf')
    failure_reason: Optional[str] = None

@dataclass(frozen=True)
class OnlineAllocation:
    search_assignments: Tuple[SearchAssignment, ...]
    executor_assignment: ExecutorAssignment
    objective_value: float
    detection_probability: float
    response_time: float
    trigger_reason: str
    status: str = 'OK'
    search_frozen: bool = False

    @property
    def allocation_sha256(self) -> str:
        payload = {'search': [(item.agent_id, item.candidate_id, item.waypoint) for item in self.search_assignments], 'executor': (self.executor_assignment.executor_id, self.executor_assignment.target_region, self.executor_assignment.source), 'objective': self.objective_value, 'detection': self.detection_probability, 'response': self.response_time if self.response_time != float('inf') else 'inf', 'status': self.status, 'search_frozen': self.search_frozen}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

@dataclass(frozen=True)
class CurrentAssignment:
    agent_id: int
    old_waypoint: Vector3 | None
    new_waypoint: Vector3
    switch_reason: str
    timestamp: int

@dataclass(frozen=True)
class InitialBSERAllocation:
    allocation: OnlineAllocation
    waypoint_updates: Tuple[CurrentAssignment, ...]

@dataclass(frozen=True)
class BSERActionAssignment:
    step: int
    replanned: bool
    events: Tuple[BSEREvent, ...]
    allocation: OnlineAllocation
    waypoint_updates: Tuple[CurrentAssignment, ...]
    decision_reason: str
    event_detection: EventDetection
    diagnostics: 'OnlineStepDiagnostics | None' = None

@dataclass(frozen=True)
class OnlineStepDiagnostics:
    step: int
    mechanism_version: str
    detected_events: Tuple[str, ...]
    optimizer_invoked: bool
    allocation_scope: str
    old_objective: float
    proposed_objective: float
    objective_gain: float
    accepted: bool
    accept_reason: str
    reject_reason: str
    affected_agent_ids: Tuple[int, ...]
    obstacle_route_impacted: bool
    old_waypoints: Tuple[Tuple[int, Vector3], ...]
    proposed_waypoints: Tuple[Tuple[int, Vector3], ...]
    switch_distance_by_agent: Tuple[Tuple[int, float], ...]
    executor_target_source: str
