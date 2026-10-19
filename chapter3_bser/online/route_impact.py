from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Tuple
import numpy as np
from chapter3_bser.online.types import OnlineAllocation
from core.mapping.planning_state import PlanningStateView
from core.mapping.travel_cost_service import TravelCostService

@dataclass(frozen=True)
class RouteCheck:
    agent_id: int
    path_intersection_count: int
    corridor_intersection_count: int
    corridor_probability_mass: float
    path_became_unreachable: bool
    connected_component_changed: bool
    planning_cost_relative_change: float
    route_impacted: bool

@dataclass(frozen=True)
class RouteImpactResult:
    route_impacted: bool
    affected_agent_ids: Tuple[int, ...]
    affected_searcher_ids: Tuple[int, ...]
    executor_route_impacted: bool
    new_occupied_cell_count: int
    new_occupied_probability_mass: float
    checks: Tuple[RouteCheck, ...]

class RouteImpactEvaluator:

    def __init__(self, *, planning_cost_threshold: float=0.15, corridor_mass_threshold: float=0.2):
        self.planning_cost_threshold = float(planning_cost_threshold)
        self.corridor_mass_threshold = float(corridor_mass_threshold)

    @staticmethod
    def _corridor(indices: Tuple[int, ...], shape: Tuple[int, int, int]) -> set[int]:
        output: set[int] = set()
        for index in indices:
            cell = np.unravel_index(int(index), shape)
            for offset in np.ndindex((3, 3, 3)):
                other = tuple((cell[axis] + offset[axis] - 1 for axis in range(3)))
                if all((0 <= other[axis] < shape[axis] for axis in range(3))):
                    output.add(int(np.ravel_multi_index(other, shape)))
        return output

    @staticmethod
    def _relative_change(current: float, previous: float) -> float:
        if not math.isfinite(current):
            return math.inf
        if not math.isfinite(previous) or previous <= 1e-12:
            return 0.0
        return (float(current) - float(previous)) / float(previous)

    def evaluate(self, previous: PlanningStateView, current: PlanningStateView, allocation: OnlineAllocation) -> RouteImpactResult:
        previous_mask = np.asarray(previous.occupancy.occupied_mask, dtype=np.bool_)
        current_mask = np.asarray(current.occupancy.occupied_mask, dtype=np.bool_)
        new_mask = current_mask & ~previous_mask
        new_indices = set((int(value) for value in np.flatnonzero(new_mask)))
        previous_risk = np.asarray(previous.occupancy.occupancy_probability, dtype=np.float64)
        current_risk = np.asarray(current.occupancy.occupancy_probability, dtype=np.float64)
        risk_delta = np.maximum(current_risk - previous_risk, 0.0)
        service = TravelCostService(current)
        checks = []
        routes = [(item.agent_id, item.waypoint, item.path_cell_indices, item.planning_cost) for item in allocation.search_assignments]
        executor = allocation.executor_assignment
        routes.append((executor.executor_id, executor.target_region, executor.path_cell_indices, executor.planning_cost))
        for (agent_id, target, raw_path, old_cost) in routes:
            path = tuple((int(value) for value in raw_path))
            path_set = set(path)
            corridor = self._corridor(path, current.grid.shape)
            intersection = len(path_set & new_indices)
            corridor_intersection = len(corridor & new_indices)
            corridor_mass = float(np.sum(risk_delta[list(corridor)], dtype=np.float64)) if corridor else 0.0
            agent = current.agents[agent_id]
            query = service.query(agent.position, target, agent)
            unreachable = not bool(query.reachable)
            component_changed = False
            if path:
                previous_labels = np.asarray(previous.planning_graph.component_labels)
                current_labels = np.asarray(current.planning_graph.component_labels)
                endpoints = (path[0], path[-1])
                component_changed = any((int(previous_labels[index]) != int(current_labels[index]) for index in endpoints))
            relative = self._relative_change(float(query.planning_cost), float(old_cost))
            impacted = bool(intersection > 0 or corridor_intersection > 0 or unreachable or component_changed or (relative >= self.planning_cost_threshold) or (corridor_mass >= self.corridor_mass_threshold))
            checks.append(RouteCheck(agent_id=int(agent_id), path_intersection_count=intersection, corridor_intersection_count=corridor_intersection, corridor_probability_mass=corridor_mass, path_became_unreachable=unreachable, connected_component_changed=component_changed, planning_cost_relative_change=relative, route_impacted=impacted))
        affected = tuple(sorted((item.agent_id for item in checks if item.route_impacted)))
        searcher_set = set(current.searcher_ids)
        affected_searchers = tuple((agent_id for agent_id in affected if agent_id in searcher_set))
        executor_impacted = current.executor_id in affected
        return RouteImpactResult(route_impacted=bool(affected), affected_agent_ids=affected, affected_searcher_ids=affected_searchers, executor_route_impacted=executor_impacted, new_occupied_cell_count=len(new_indices), new_occupied_probability_mass=float(np.sum(current_risk[list(new_indices)], dtype=np.float64)) if new_indices else 0.0, checks=tuple(checks))
