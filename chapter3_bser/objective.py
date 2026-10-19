from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Tuple
import numpy as np
from chapter3_bser.detection_model import detection_probability
from chapter3_bser.types import SearchCandidate
from chapter3_bser.types import StandbyCandidate
from core.mapping.planning_state import PlanningStateView
from core.mapping.travel_cost_service import TravelCostService

@dataclass(frozen=True)
class ObjectiveContext:
    state: PlanningStateView
    candidates: Tuple[SearchCandidate, ...]
    standby_candidates: Tuple[StandbyCandidate, ...]
    belief: np.ndarray
    detection_by_id: Dict[str, np.ndarray]
    response_weight_by_id: Dict[str, np.ndarray]
    response_time_by_id: Dict[str, np.ndarray]
    epsilon: float

@dataclass(frozen=True)
class ResponseDiagnostics:
    total_detected_mass: float
    reachable_detected_mass: float
    unreachable_detected_mass: float
    unreachable_detected_mass_ratio: float
    conditional_reachable_response_time: float
    maximum_reachable_response_time: float
    response_defined: bool
    all_detected_mass_reachable: bool

def build_objective_context(state: PlanningStateView, candidates: Sequence[SearchCandidate], standby_candidates: Sequence[StandbyCandidate], config: dict) -> ObjectiveContext:
    belief = np.asarray(state.target_belief.probabilities, dtype=np.float64).copy()
    belief.setflags(write=False)
    detection = {candidate.candidate_id: detection_probability(candidate, state, config) for candidate in candidates}
    service = TravelCostService(state)
    executor = state.agents[state.executor_id]
    tau = float(config['objective']['tau_executor'])
    response_time = {}
    response_weight = {}
    for standby in standby_candidates:
        times = np.asarray(service.travel_times_from(standby.waypoint, executor), dtype=np.float64)
        weights = np.zeros(times.shape, dtype=np.float64)
        finite = np.isfinite(times)
        weights[finite] = np.exp(-times[finite] / tau)
        weights = np.clip(weights, 0.0, 1.0)
        weights.setflags(write=False)
        response_time[standby.candidate_id] = times
        response_weight[standby.candidate_id] = weights
    return ObjectiveContext(state=state, candidates=tuple(candidates), standby_candidates=tuple(standby_candidates), belief=belief, detection_by_id=detection, response_weight_by_id=response_weight, response_time_by_id=response_time, epsilon=float(config['objective']['epsilon']))

def cell_detection_probability(selected: Iterable[SearchCandidate], context: ObjectiveContext) -> np.ndarray:
    complement = np.ones(context.belief.shape, dtype=np.float64)
    for candidate in selected:
        complement *= 1.0 - context.detection_by_id[candidate.candidate_id]
    result = np.clip(1.0 - complement, 0.0, 1.0)
    result.setflags(write=False)
    return result

def evaluate_objective(selected: Iterable[SearchCandidate], standby: Optional[StandbyCandidate], context: ObjectiveContext, *, search_only: bool=False) -> float:
    detection = cell_detection_probability(selected, context)
    if search_only:
        weight = 1.0
    else:
        if standby is None:
            return 0.0
        weight = context.response_weight_by_id[standby.candidate_id]
    value = float(np.sum(context.belief * detection * weight, dtype=np.float64))
    return max(0.0, value)

def marginal_gain(selected: Iterable[SearchCandidate], candidate: SearchCandidate, standby: Optional[StandbyCandidate], context: ObjectiveContext, *, search_only: bool=False) -> float:
    current = tuple(selected)
    return evaluate_objective(current + (candidate,), standby, context, search_only=search_only) - evaluate_objective(current, standby, context, search_only=search_only)

def expected_detection_probability(selected: Iterable[SearchCandidate], context: ObjectiveContext) -> float:
    return float(np.sum(context.belief * cell_detection_probability(selected, context), dtype=np.float64))

def response_diagnostics(selected: Iterable[SearchCandidate], standby: StandbyCandidate, context: ObjectiveContext) -> ResponseDiagnostics:
    detection = cell_detection_probability(selected, context)
    mass = context.belief * detection
    times = context.response_time_by_id[standby.candidate_id]
    finite = np.isfinite(times)
    total = float(np.sum(mass, dtype=np.float64))
    reachable = float(np.sum(mass[finite], dtype=np.float64))
    unreachable = max(0.0, total - reachable)
    defined = reachable > context.epsilon
    if defined:
        numerator = float(np.sum(mass[finite] * times[finite], dtype=np.float64))
        conditional = numerator / (reachable + context.epsilon)
        positive_reachable = finite & (mass > context.epsilon)
        maximum = float(np.max(times[positive_reachable])) if np.any(positive_reachable) else 0.0
    else:
        conditional = float('inf')
        maximum = float('inf')
    return ResponseDiagnostics(total_detected_mass=total, reachable_detected_mass=reachable, unreachable_detected_mass=unreachable, unreachable_detected_mass_ratio=unreachable / (total + context.epsilon), conditional_reachable_response_time=conditional, maximum_reachable_response_time=maximum, response_defined=defined, all_detected_mass_reachable=unreachable <= context.epsilon)
