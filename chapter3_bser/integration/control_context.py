from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
Vector3 = Tuple[float, float, float]

@dataclass(frozen=True)
class AgentAssignmentContextV1:
    """One agent's read-only, action-free high-level assignment."""
    agent_id: int
    role: str
    assignment_kind: str
    assignment_id: str
    final_waypoint: Vector3
    planned_path: Tuple[Vector3, ...]
    tracking_waypoint: Vector3
    hold_position: Vector3
    hold_state: bool
    reachable: bool
    execution_request: bool

@dataclass(frozen=True)
class ExecutorAssignmentContextV1:
    """Executor-specific view retained separately for diagnostics and gating."""
    executor_id: int
    source: str
    target_region: Vector3
    planned_path: Tuple[Vector3, ...]
    tracking_waypoint: Vector3
    hold_position: Vector3
    hold_state: bool
    reachable: bool
    execution_request: bool

@dataclass(frozen=True)
class BSERControlContextV1:
    """Versioned immutable contract at the BSER/RMADDPG boundary.

    The context contains navigation intent only.  It deliberately has no
    velocity, acceleration, action, reward, or neural-network fields.
    """
    schema_version: str
    allocation_version: str
    allocation_hash: str
    step: int
    mission_phase: str
    agent_assignments: Tuple[AgentAssignmentContextV1, ...]
    executor_assignment: ExecutorAssignmentContextV1
    decision_reason: str
