from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
from typing import Optional
from typing import Tuple
import numpy as np

@dataclass(frozen=True)
class SearchCandidate:
    agent_id: int
    candidate_id: str
    waypoint: Tuple[float, float, float]
    path_points: np.ndarray
    path_cell_indices: np.ndarray
    path_length: float
    planning_cost: float
    physical_travel_time: float
    source: str

    @property
    def key(self):
        return (self.agent_id, self.candidate_id, self.waypoint)

    @property
    def travel_time(self):
        return self.physical_travel_time

@dataclass(frozen=True)
class StandbyCandidate:
    candidate_id: str
    waypoint: Tuple[float, float, float]
    path_points: np.ndarray
    path_cell_indices: np.ndarray
    path_length: float
    planning_cost: float
    physical_travel_time: float
    source: str

    @property
    def key(self):
        return (self.candidate_id, self.waypoint)

    @property
    def travel_time(self):
        return self.physical_travel_time

@dataclass(frozen=True)
class CandidateGenerationResult:
    search_candidates: Tuple[SearchCandidate, ...]
    standby_candidates: Tuple[StandbyCandidate, ...]
    search_count_by_agent: Mapping[int, int]
    unreachable_search_count: int
    unreachable_standby_count: int
    reasons: Tuple[str, ...]

@dataclass(frozen=True)
class SolverResult:
    solver: str
    selected: Tuple[SearchCandidate, ...]
    standby: Optional[StandbyCandidate]
    objective: float
    combination_count: int = 0
    status: str = 'OK'
