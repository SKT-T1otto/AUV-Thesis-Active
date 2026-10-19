from __future__ import annotations
from dataclasses import dataclass
import hashlib
import heapq
import math
from typing import Iterable
from typing import Optional
from typing import Tuple
import numpy as np
from core.mapping.planning_graph import PlanningConnectorView
from core.mapping.planning_state import PlanningAgentView
from core.mapping.planning_state import PlanningStateView

def _locked(value, *, dtype, shape=None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    result = np.frombuffer(array.tobytes(order='C'), dtype=array.dtype)
    result = result.reshape(array.shape if shape is None else shape)
    result.setflags(write=False)
    return result

@dataclass(frozen=True)
class PathQueryResult:
    reachable: bool
    planning_cost: float
    physical_travel_time: float
    path_length: float
    unknown_fraction: float
    occupancy_risk: float
    path_points: np.ndarray
    path_cell_indices: np.ndarray
    failure_reason: Optional[str]

@dataclass(frozen=True)
class SingleSourceTravelResult:
    planning_cost_by_cell: np.ndarray
    physical_time_by_cell: np.ndarray
    predecessor_by_cell: np.ndarray
    reachable_mask: np.ndarray
    path_tree_sha256: str

class TravelCostService:
    """A* and Dijkstra using only the frozen graph supplied in the state."""

    def __init__(self, state: PlanningStateView):
        self.state = state
        self.graph = state.planning_graph
        self._centers = np.asarray(self.graph.cell_centers, dtype=np.float64)
        self._labels = np.asarray(self.graph.component_labels, dtype=np.int64)
        self._valid = np.asarray(self.graph.valid_mask, dtype=np.bool_)

    def _center_connector(self, point) -> Optional[PlanningConnectorView]:
        value = np.asarray(tuple(point), dtype=np.float64).reshape(3)
        squared = np.einsum('ij,ij->i', self._centers - value, self._centers - value)
        index = int(np.argmin(squared))
        if float(squared[index]) > 1e-18 or not bool(self._valid[index]):
            return None
        component = int(self._labels[index])
        if component < 0:
            return None
        return PlanningConnectorView(index, component, 0.0, 0.0)

    def _connectors(self, point: Iterable[float], role: str) -> Tuple[PlanningConnectorView, ...]:
        value = tuple((float(item) for item in np.asarray(tuple(point), dtype=np.float64).reshape(3)))
        role_class = 'executor' if str(role).lower().startswith('exec') else 'searcher'
        for endpoint in self.graph.endpoint_connectors:
            endpoint_class = 'executor' if endpoint.role.lower().startswith('exec') else 'searcher'
            if endpoint_class == role_class and np.allclose(endpoint.point, value, atol=1e-12, rtol=0.0):
                return endpoint.connectors
        center = self._center_connector(value)
        return () if center is None else (center,)

    def _known_invalid_endpoint(self, point: Iterable[float], role: str) -> bool:
        value = tuple((float(item) for item in np.asarray(tuple(point), dtype=np.float64).reshape(3)))
        role_class = 'executor' if str(role).lower().startswith('exec') else 'searcher'
        for endpoint in self.graph.endpoint_connectors:
            endpoint_class = 'executor' if endpoint.role.lower().startswith('exec') else 'searcher'
            if endpoint_class == role_class and np.allclose(endpoint.point, value, atol=1e-12, rtol=0.0):
                return not endpoint.point_valid
        delta = self._centers - np.asarray(value, dtype=np.float64)
        squared = np.einsum('ij,ij->i', delta, delta)
        index = int(np.argmin(squared))
        return float(squared[index]) <= 1e-18 and (not bool(self._valid[index]))

    def _heuristic(self, source: int, goal: int, role: str) -> float:
        delta = np.abs(self._centers[int(goal)].astype(np.float32) - self._centers[int(source)].astype(np.float32))
        horizontal = float(np.linalg.norm(delta[:2]).astype(np.float32))
        if str(role).lower().startswith('exec'):
            divisor = self.graph.executor_horizontal_speed_divisor
            lookup = self.graph.executor_vertical_cost_lookup
        else:
            divisor = self.graph.searcher_horizontal_speed_divisor
            lookup = self.graph.searcher_vertical_cost_lookup
        vertical_delta = float(delta[2])
        vertical = min(lookup, key=lambda item: abs(item[0] - vertical_delta))[1] if lookup else vertical_delta * self.graph.time_rates_for_role(role)[1]
        return horizontal / divisor + vertical

    def _astar(self, start: int, goal: int, role: str):
        if start == goal:
            return (True, (start,), 0.0, 0.0)
        adjacency = self.graph.adjacency_for_role(role)
        queue = [(self._heuristic(start, goal, role), 0.0, int(start))]
        costs = {int(start): 0.0}
        physical = {int(start): 0.0}
        parents = {}
        while queue:
            (_, cost, cell) = heapq.heappop(queue)
            if cost > costs.get(cell, math.inf) + 1e-12:
                continue
            if cell == goal:
                break
            for edge in adjacency[cell]:
                new_cost = cost + edge.planning_cost
                if new_cost + 1e-12 < costs.get(edge.destination, math.inf):
                    costs[edge.destination] = new_cost
                    physical[edge.destination] = physical[cell] + edge.physical_travel_time
                    parents[edge.destination] = cell
                    heapq.heappush(queue, (new_cost + self._heuristic(edge.destination, goal, role), new_cost, edge.destination))
        if goal not in costs:
            return (False, (), math.inf, math.inf)
        cells = [int(goal)]
        while cells[-1] != int(start):
            cells.append(parents[cells[-1]])
        cells.reverse()
        return (True, tuple(cells), float(costs[goal]), float(physical[goal]))

    @staticmethod
    def _failure(reason: str) -> PathQueryResult:
        return PathQueryResult(False, math.inf, math.inf, math.inf, 0.0, 0.0, _locked([], dtype=np.float64, shape=(0, 3)), _locked([], dtype=np.int64), str(reason))

    def query(self, start: Iterable[float], goal: Iterable[float], agent: PlanningAgentView) -> PathQueryResult:
        start_array = np.asarray(tuple(start), dtype=np.float64).reshape(3)
        goal_array = np.asarray(tuple(goal), dtype=np.float64).reshape(3)
        role = agent.role
        if self._known_invalid_endpoint(start_array, role):
            return self._failure('invalid_start')
        if self._known_invalid_endpoint(goal_array, role):
            return self._failure('invalid_goal')
        starts = self._connectors(start_array, role)
        goals = self._connectors(goal_array, role)
        if not starts:
            return self._failure('no_start_connector')
        if not goals:
            return self._failure('no_goal_connector')
        if np.allclose(start_array, goal_array, atol=1e-12, rtol=0.0):
            return PathQueryResult(True, 0.0, 0.0, 0.0, 0.0, 0.0, _locked(start_array.reshape(1, 3), dtype=np.float64), _locked([], dtype=np.int64), None)
        best = None
        for start_connector in starts:
            for goal_connector in goals:
                if start_connector.component_id != goal_connector.component_id:
                    continue
                (reachable, cells, grid_cost, grid_physical) = self._astar(start_connector.cell_index, goal_connector.cell_index, role)
                if not reachable:
                    continue
                total = start_connector.planning_cost + grid_cost + goal_connector.planning_cost
                route_key = (total, start_connector.cell_index, goal_connector.cell_index)
                if best is None or route_key < best[0]:
                    best = (route_key, start_connector, goal_connector, cells, grid_physical)
        if best is None:
            return self._failure('disconnected_endpoint_components')
        (route_key, start_connector, goal_connector, cells, grid_physical) = best
        points = [start_array]
        for cell in cells:
            center = self._centers[cell]
            if np.linalg.norm(center - points[-1]) > 1e-12:
                points.append(center)
        if np.linalg.norm(goal_array - points[-1]) > 1e-12:
            points.append(goal_array)
        point_array = np.asarray(points, dtype=np.float64)
        path_length = float(np.linalg.norm(np.diff(point_array, axis=0), axis=1).sum()) if len(points) > 1 else 0.0
        indices = np.asarray(cells, dtype=np.int64)
        unknown = np.asarray(self.state.occupancy.unknown_mask, dtype=np.bool_)
        occupancy = np.asarray(self.state.occupancy.occupancy_probability, dtype=np.float64)
        return PathQueryResult(reachable=True, planning_cost=float(route_key[0]), physical_travel_time=float(start_connector.physical_travel_time + grid_physical + goal_connector.physical_travel_time), path_length=path_length, unknown_fraction=float(np.mean(unknown[indices])) if indices.size else 0.0, occupancy_risk=float(np.mean(occupancy[indices])) if indices.size else 0.0, path_points=_locked(point_array, dtype=np.float64), path_cell_indices=_locked(indices, dtype=np.int64), failure_reason=None)

    def single_source(self, start: Iterable[float], agent: PlanningAgentView) -> SingleSourceTravelResult:
        count = self._centers.shape[0]
        costs = np.full(count, np.inf, dtype=np.float64)
        physical = np.full(count, np.inf, dtype=np.float64)
        predecessor = np.full(count, -1, dtype=np.int64)
        connectors = self._connectors(start, agent.role)
        queue = []
        for connector in connectors:
            index = connector.cell_index
            if connector.planning_cost + 1e-12 < costs[index]:
                costs[index] = connector.planning_cost
                physical[index] = connector.physical_travel_time
                heapq.heappush(queue, (connector.planning_cost, index))
        adjacency = self.graph.adjacency_for_role(agent.role)
        while queue:
            (cost, cell) = heapq.heappop(queue)
            if cost > costs[cell] + 1e-12:
                continue
            for edge in adjacency[cell]:
                new_cost = cost + edge.planning_cost
                if new_cost + 1e-12 < costs[edge.destination]:
                    costs[edge.destination] = new_cost
                    physical[edge.destination] = physical[cell] + edge.physical_travel_time
                    predecessor[edge.destination] = cell
                    heapq.heappush(queue, (new_cost, edge.destination))
        reachable = np.isfinite(costs)
        digest = hashlib.sha256()
        digest.update(costs.tobytes(order='C'))
        digest.update(physical.tobytes(order='C'))
        digest.update(predecessor.tobytes(order='C'))
        return SingleSourceTravelResult(_locked(costs, dtype=np.float64), _locked(physical, dtype=np.float64), _locked(predecessor, dtype=np.int64), _locked(reachable, dtype=np.bool_), digest.hexdigest())

    def travel_times_from(self, start: Iterable[float], agent: PlanningAgentView) -> np.ndarray:
        return self.single_source(start, agent).physical_time_by_cell
