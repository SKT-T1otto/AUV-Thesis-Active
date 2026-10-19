from __future__ import annotations
import math
from dataclasses import replace
import numpy as np
from chapter3_bser.online.types import ExecutorAssignment
from core.mapping.planning_state import PlanningStateView
from core.mapping.travel_cost_service import TravelCostService

def _path(value) -> tuple[tuple[float, float, float], ...]:
    array = np.asarray(value, dtype=np.float64).reshape(-1, 3)
    return tuple((tuple((float(x) for x in row)) for row in array))

class ExecutionManager:

    @staticmethod
    def _assignment(state: PlanningStateView, target_region, *, source: str) -> ExecutorAssignment:
        executor = state.agents[state.executor_id]
        target = tuple((float(value) for value in target_region))
        result = TravelCostService(state).query(executor.position, target, executor)
        return ExecutorAssignment(executor_id=state.executor_id, target_region=target, path=_path(result.path_points) if result.reachable else (), estimated_arrival_time=float(result.physical_travel_time) if result.reachable else math.inf, source=str(source), reachable=bool(result.reachable), path_cell_indices=tuple((int(value) for value in result.path_cell_indices)), planning_cost=float(result.planning_cost), failure_reason=result.failure_reason)

    def assign_belief_peak(self, state: PlanningStateView, *, source: str='belief_peak') -> ExecutorAssignment:
        target_region = tuple((float(value) for value in state.grid.cell_centers[state.target_belief.peak_index]))
        return self._assignment(state, target_region, source=source)

    def assign_standby(self, state: PlanningStateView, standby) -> ExecutorAssignment:
        if standby is None:
            return self.assign_belief_peak(state, source='fallback_belief_peak')
        return ExecutorAssignment(executor_id=state.executor_id, target_region=tuple((float(value) for value in standby.waypoint)), path=_path(standby.path_points), estimated_arrival_time=float(standby.physical_travel_time), source=f'standby:{standby.candidate_id}', reachable=True, path_cell_indices=tuple((int(value) for value in standby.path_cell_indices)), planning_cost=float(standby.planning_cost))

    def assign_public_handoff(self, state: PlanningStateView, target_region) -> ExecutorAssignment:
        return self._assignment(state, target_region, source='PUBLIC_HANDOFF_TARGET')

    def preserve_current(self, state: PlanningStateView, current: ExecutorAssignment) -> ExecutorAssignment:
        return self._assignment(state, current.target_region, source='CURRENT_VALID_ROUTE')

    def assign_reachable_belief(self, state: PlanningStateView) -> ExecutorAssignment:
        peak = self.assign_belief_peak(state, source='BELIEF_PEAK_FALLBACK')
        if peak.reachable:
            return peak
        probabilities = np.asarray(state.target_belief.probabilities, dtype=np.float64)
        ranking = np.lexsort((np.arange(probabilities.size), -probabilities))
        peak_index = int(state.target_belief.peak_index)
        for index in ranking:
            if int(index) == peak_index:
                continue
            candidate = self._assignment(state, state.grid.cell_centers[int(index)], source='NEAREST_REACHABLE_BELIEF_SUPPORT')
            if candidate.reachable:
                return candidate
        return replace(peak, source='NO_VALID_ROUTE')

    def assign_after_public_handoff(self, state: PlanningStateView, target_region, current: ExecutorAssignment) -> ExecutorAssignment:
        if target_region is not None:
            public = self.assign_public_handoff(state, target_region)
            if public.reachable:
                return public
        preserved = self.preserve_current(state, current)
        if preserved.reachable:
            return preserved
        return self.assign_reachable_belief(state)
