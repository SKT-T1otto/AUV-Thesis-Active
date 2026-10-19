from __future__ import annotations
import math
from dataclasses import replace
from typing import Mapping
import numpy as np
from chapter3_bser.candidate_generator import generate_candidates
from chapter3_bser.candidate_generator import generate_search_candidates
from chapter3_bser.candidate_generator import generate_standby_candidates
from chapter3_bser.config import load_bser_phase1a1_config
from chapter3_bser.greedy_solver import solve_joint_greedy
from chapter3_bser.objective import build_objective_context
from chapter3_bser.objective import evaluate_objective
from chapter3_bser.objective import expected_detection_probability
from chapter3_bser.objective import response_diagnostics
from chapter3_bser.online.execution_manager import ExecutionManager
from chapter3_bser.online.types import OnlineAllocation
from chapter3_bser.online.types import SearchAssignment
from chapter3_bser.types import SearchCandidate
from chapter3_bser.types import StandbyCandidate
from core.mapping.planning_state import PlanningStateView
from core.mapping.travel_cost_service import TravelCostService

def _path(value) -> tuple[tuple[float, float, float], ...]:
    array = np.asarray(value, dtype=np.float64).reshape(-1, 3)
    return tuple((tuple((float(x) for x in row)) for row in array))

class BSEROnlineAllocator:

    def __init__(self, phase1a1_config: Mapping | None=None):
        self.config = dict(phase1a1_config or load_bser_phase1a1_config())
        self.execution = ExecutionManager()

    def _solve_candidates(self, candidates, standby_candidates, context):
        """Default frozen solver; chapter experiments may override ranking only."""
        return solve_joint_greedy(candidates, standby_candidates, context)

    def _generate_candidates(self, state):
        """Default candidate assembly; independent experiments may filter routes."""
        return generate_candidates(state, self.config)

    def allocate(self, state: PlanningStateView, *, trigger_reason: str='online') -> OnlineAllocation:
        generated = self._generate_candidates(state)
        observer = getattr(self, '_audit_candidate_generation_observer', None)
        if observer is not None:
            observer(state, generated.search_candidates, generated.standby_candidates, scope='full', agent_ids=state.searcher_ids)
        if not generated.search_candidates or not generated.standby_candidates:
            executor = self.execution.assign_belief_peak(state, source='no_feasible_candidate_fallback')
            return OnlineAllocation((), executor, 0.0, 0.0, math.inf, trigger_reason, 'NO_FEASIBLE_CANDIDATES')
        context = build_objective_context(state, generated.search_candidates, generated.standby_candidates, self.config)
        solved = self._solve_candidates(generated.search_candidates, generated.standby_candidates, context)
        if solved.standby is None:
            executor = self.execution.assign_belief_peak(state, source='no_standby_fallback')
            return OnlineAllocation((), executor, 0.0, 0.0, math.inf, trigger_reason, solved.status)
        search = tuple((SearchAssignment(candidate.agent_id, candidate.candidate_id, tuple((float(value) for value in candidate.waypoint)), _path(candidate.path_points), float(candidate.physical_travel_time), tuple((int(value) for value in candidate.path_cell_indices)), float(candidate.planning_cost)) for candidate in solved.selected))
        diagnostics = response_diagnostics(solved.selected, solved.standby, context)
        return OnlineAllocation(search_assignments=search, executor_assignment=self.execution.assign_standby(state, solved.standby), objective_value=float(solved.objective), detection_probability=float(expected_detection_probability(solved.selected, context)), response_time=float(diagnostics.conditional_reachable_response_time), trigger_reason=str(trigger_reason), status=solved.status)

    def reassign_after_target_found(self, state: PlanningStateView, *, trigger_reason: str='TARGET_FOUND') -> OnlineAllocation:
        executor = self.execution.assign_belief_peak(state, source='target_found_belief_peak')
        return OnlineAllocation((), executor, 0.0, 0.0, executor.estimated_arrival_time, trigger_reason, 'OK' if executor.reachable else 'EXECUTOR_UNREACHABLE', True)

    def reassign_after_target_received(self, state: PlanningStateView, current: OnlineAllocation, public_target, *, trigger_reason: str='EXECUTOR_TARGET_RECEIVED') -> OnlineAllocation:
        executor = self.execution.assign_after_public_handoff(state, public_target, current.executor_assignment)
        return OnlineAllocation(search_assignments=current.search_assignments, executor_assignment=executor, objective_value=current.objective_value, detection_probability=current.detection_probability, response_time=executor.estimated_arrival_time, trigger_reason=trigger_reason, status='OK' if executor.reachable else 'EXECUTOR_UNREACHABLE', search_frozen=True)

    def replan_executor_to_public_target(self, state: PlanningStateView, current: OnlineAllocation, public_target, *, trigger_reason: str='EXECUTOR_PUBLIC_TARGET_REPLAN') -> tuple[OnlineAllocation, bool, str]:
        """Replan only the executor while preserving the public handoff target.

        The public target remains installed even when the current planning graph
        cannot reach it. In that case the low-level fixed adapter holds position
        and a later refreshed planning state may produce a feasible path.
        """
        if public_target is None:
            return (current, False, 'ATOMIC_REJECT_PUBLIC_TARGET_UNAVAILABLE')
        executor = self.execution.assign_public_handoff(state, public_target)
        allocation = OnlineAllocation(search_assignments=current.search_assignments, executor_assignment=executor, objective_value=current.objective_value, detection_probability=current.detection_probability, response_time=executor.estimated_arrival_time, trigger_reason=trigger_reason, status='OK' if executor.reachable else 'EXECUTOR_UNREACHABLE', search_frozen=True)
        reason = 'EXECUTOR_PUBLIC_TARGET_REPLAN' if executor.reachable else 'EXECUTOR_PUBLIC_TARGET_LOCKED_UNREACHABLE'
        return (allocation, True, reason)

    def reassign_invalid_executor(self, state: PlanningStateView, current: OnlineAllocation, *, trigger_reason: str='EXECUTOR_INVALID') -> OnlineAllocation:
        executor = self.execution.assign_reachable_belief(state)
        return OnlineAllocation(current.search_assignments, executor, current.objective_value, current.detection_probability, executor.estimated_arrival_time, trigger_reason, 'OK' if executor.reachable else 'EXECUTOR_UNREACHABLE', True)

    def _partial_search_candidates(self, state: PlanningStateView, affected: set[int]):
        generation = self.config['candidate_generation']
        subset = replace(state, searcher_ids=tuple(sorted(affected)))
        (candidates, _, _) = generate_search_candidates(subset, TravelCostService(state), k_search=int(generation['k_search_exact']), minimum_separation=float(generation['minimum_separation']), maximum_travel_time=float(generation.get('maximum_physical_travel_time', generation.get('maximum_travel_time'))))
        return candidates

    @staticmethod
    def _path_length(path) -> float:
        points = np.asarray(path, dtype=np.float64).reshape(-1, 3)
        if len(points) < 2:
            return 0.0
        return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

    @classmethod
    def _frozen_search_candidate(cls, item: SearchAssignment) -> SearchCandidate:
        path = np.asarray(item.path, dtype=np.float64).reshape(-1, 3)
        indices = np.asarray(item.path_cell_indices, dtype=np.int64)
        path.setflags(write=False)
        indices.setflags(write=False)
        return SearchCandidate(agent_id=int(item.agent_id), candidate_id=str(item.candidate_id), waypoint=tuple((float(value) for value in item.waypoint)), path_points=path, path_cell_indices=indices, path_length=cls._path_length(path), planning_cost=float(item.planning_cost), physical_travel_time=float(item.physical_travel_time), source='current_assignment')

    @classmethod
    def _frozen_standby_candidate(cls, current: OnlineAllocation) -> StandbyCandidate:
        item = current.executor_assignment
        path = np.asarray(item.path, dtype=np.float64).reshape(-1, 3)
        indices = np.asarray(item.path_cell_indices, dtype=np.int64)
        path.setflags(write=False)
        indices.setflags(write=False)
        return StandbyCandidate(candidate_id='partial_current_executor', waypoint=tuple((float(value) for value in item.target_region)), path_points=path, path_cell_indices=indices, path_length=cls._path_length(path), planning_cost=float(item.planning_cost), physical_travel_time=float(item.estimated_arrival_time), source='current_assignment')

    @staticmethod
    def _search_assignment(candidate: SearchCandidate) -> SearchAssignment:
        return SearchAssignment(candidate.agent_id, candidate.candidate_id, tuple((float(value) for value in candidate.waypoint)), _path(candidate.path_points), float(candidate.physical_travel_time), tuple((int(value) for value in candidate.path_cell_indices)), float(candidate.planning_cost))

    def allocate_partial(self, state: PlanningStateView, current: OnlineAllocation, *, affected_searcher_ids=(), executor_affected: bool=False, trigger_reason: str) -> tuple[OnlineAllocation, bool, str]:
        affected = {int(value) for value in affected_searcher_ids}
        old_search = {item.agent_id: item for item in current.search_assignments}
        if not affected.issubset(set(state.searcher_ids)):
            return (current, False, 'ATOMIC_REJECT_UNKNOWN_SEARCHER')
        affected_candidates = ()
        if affected:
            affected_candidates = self._partial_search_candidates(state, affected)
            observer = getattr(self, '_audit_candidate_generation_observer', None)
            if observer is not None:
                observer(state, affected_candidates, None, scope='partial_search', agent_ids=tuple(sorted(affected)))
            for agent_id in sorted(affected):
                pool = [candidate for candidate in affected_candidates if candidate.agent_id == agent_id]
                if not pool:
                    return (current, False, 'ATOMIC_REJECT_MISSING_SEARCH_ROUTE')
        frozen_candidates = tuple((self._frozen_search_candidate(item) for (agent_id, item) in sorted(old_search.items()) if agent_id not in affected))
        search_candidates = tuple(affected_candidates) + frozen_candidates
        generation = self.config['candidate_generation']
        if executor_affected:
            (standby_candidates, _, _) = generate_standby_candidates(state, TravelCostService(state), k_standby=int(generation['k_standby_exact']))
            observer = getattr(self, '_audit_candidate_generation_observer', None)
            if observer is not None:
                observer(state, (), standby_candidates, scope='partial_standby', agent_ids=())
        else:
            standby_candidates = (self._frozen_standby_candidate(current),)
        if not search_candidates or not standby_candidates:
            return (current, False, 'ATOMIC_REJECT_MISSING_LOCAL_CANDIDATES')
        context = build_objective_context(state, search_candidates, standby_candidates, self.config)
        solved = self._solve_candidates(search_candidates, standby_candidates, context)
        if solved.standby is None:
            return (current, False, 'ATOMIC_REJECT_MISSING_STANDBY_ROUTE')
        selected_by_agent = {candidate.agent_id: candidate for candidate in solved.selected}
        if any((agent_id not in selected_by_agent for agent_id in affected)):
            return (current, False, 'ATOMIC_REJECT_MISSING_GREEDY_SELECTION')
        merged_assignments = []
        merged_candidates = []
        for agent_id in sorted(old_search):
            if agent_id in affected:
                candidate = selected_by_agent[agent_id]
                merged_assignments.append(self._search_assignment(candidate))
                merged_candidates.append(candidate)
            else:
                merged_assignments.append(old_search[agent_id])
                merged_candidates.append(next((candidate for candidate in frozen_candidates if candidate.agent_id == agent_id)))
        if executor_affected:
            executor = self.execution.assign_standby(state, solved.standby)
            if not executor.reachable:
                return (current, False, 'ATOMIC_REJECT_EXECUTOR_UNREACHABLE')
            objective_standby = solved.standby
        else:
            executor = current.executor_assignment
            objective_standby = standby_candidates[0]
        objective = evaluate_objective(merged_candidates, objective_standby, context)
        diagnostics = response_diagnostics(merged_candidates, objective_standby, context)
        allocation = OnlineAllocation(search_assignments=tuple(merged_assignments), executor_assignment=executor, objective_value=float(objective), detection_probability=float(expected_detection_probability(merged_candidates, context)), response_time=float(diagnostics.conditional_reachable_response_time), trigger_reason=trigger_reason, status=solved.status, search_frozen=False)
        return (allocation, True, 'ATOMIC_PARTIAL_BSER_PROPOSAL')
