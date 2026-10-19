from __future__ import annotations
import math
from typing import Mapping
import numpy as np
from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.events.event_types import EventDetection
from core.mapping.planning_state import PlanningStateView
from core.mapping.travel_cost_service import TravelCostService

class EventDetector:

    def __init__(self, config: Mapping):
        events = config['events']
        self.belief_shift_threshold = float(events['belief_shift_threshold'])
        self.obstacle_mass_threshold = float(events['obstacle_mass_threshold'])
        self.periodic_refresh_steps = int(events['periodic_refresh_steps'])
        self.waypoint_stale_distance = float(events['waypoint_stale_distance'])
        self.enable_belief_trigger = bool(events.get('enable_belief_trigger', True))
        self.enable_obstacle_trigger = bool(events.get('enable_obstacle_trigger', True))
        self.enable_target_trigger = bool(events.get('enable_target_trigger', True))
        self.mechanism_version = str(config.get('mechanism_version', 'phase1b_v1'))
        self.executor_cost_increase_threshold = float(config.get('execution', {}).get('executor_cost_increase_threshold', config.get('route_impact', {}).get('planning_cost_relative_threshold', 0.15)))
        execution_runtime = dict(config.get('execution_runtime', {}))
        self.dynamic_public_target_enabled = bool(execution_runtime.get('dynamic_public_target_enabled', False))
        self.public_target_update_distance = float(execution_runtime.get('public_target_update_distance', 0.75))
        self.public_target_update_min_steps = int(execution_runtime.get('public_target_update_min_steps', 20))
        self.defer_stale_endpoint_invalid = bool(execution_runtime.get('defer_stale_endpoint_invalid', False))

    @staticmethod
    def _relative_cost_change(current: float, installed: float) -> float:
        if not math.isfinite(current):
            return math.inf
        if not math.isfinite(installed) or installed <= 1e-12:
            return 0.0
        return (float(current) - float(installed)) / float(installed)

    @staticmethod
    def _endpoint_snapshot_present(state: PlanningStateView, point, role: str) -> bool:
        if point is None:
            return False
        value = np.asarray(tuple(point), dtype=np.float64).reshape(3)
        role_class = 'executor' if str(role).lower().startswith('exec') else 'searcher'
        for endpoint in state.planning_graph.endpoint_connectors:
            endpoint_class = 'executor' if str(endpoint.role).lower().startswith('exec') else 'searcher'
            if endpoint_class == role_class and np.allclose(endpoint.point, value, atol=1e-12, rtol=0.0):
                return True
        return False

    def detect(self, previous: PlanningStateView, current: PlanningStateView, previous_context: OnlineMissionContext | None=None, current_context: OnlineMissionContext | None=None, assignment: OnlineAllocation | None=None) -> EventDetection:
        previous_belief = np.asarray(previous.target_belief.probabilities, dtype=np.float64)
        current_belief = np.asarray(current.target_belief.probabilities, dtype=np.float64)
        if previous_belief.shape != current_belief.shape:
            raise ValueError('belief shape changed between online snapshots')
        belief_distance = float(np.sum(np.abs(current_belief - previous_belief), dtype=np.float64))
        previous_entropy = float(previous.target_belief.entropy)
        current_entropy = float(current.target_belief.entropy)
        belief_score = 0.5 * belief_distance + 0.5 * abs(current_entropy - previous_entropy)
        previous_occupied = np.asarray(previous.occupancy.occupied_mask, dtype=np.bool_)
        current_occupied = np.asarray(current.occupancy.occupied_mask, dtype=np.bool_)
        if previous_occupied.shape != current_occupied.shape:
            raise ValueError('occupancy shape changed between online snapshots')
        new_cells = current_occupied & ~previous_occupied
        current_risk = np.asarray(current.occupancy.occupancy_probability, dtype=np.float64)
        previous_risk = np.asarray(previous.occupancy.occupancy_probability, dtype=np.float64)
        new_mass = float(np.sum(current_risk[new_cells], dtype=np.float64))
        risk_change = float(np.sum(np.maximum(current_risk - previous_risk, 0.0), dtype=np.float64))
        service = TravelCostService(current)
        executor = current.agents[current.executor_id]
        executor_relative_cost_change = 0.0
        assignment_waypoints = ()
        executor_validity_evaluated = True
        executor_validity_deferred = False
        executor_invalid_reason = 'VALID'
        executor_query_failure_reason = ''
        executor_assignment_reachable = True
        executor_query_reachable = True
        executor_start_endpoint_current = True
        executor_goal_endpoint_current = True
        executor_installed_planning_cost = None
        executor_current_planning_cost = None
        executor_public_target_shift = 0.0
        public_target_updated = False
        if self.mechanism_version == 'phase1b2_corrected':
            if assignment is None:
                raise ValueError('Phase 1B.2 event detection requires the canonical assignment')
            executor_assignment = assignment.executor_assignment
            executor_assignment_reachable = bool(executor_assignment.reachable)
            executor_installed_planning_cost = float(executor_assignment.planning_cost)
            query_target = executor_assignment.target_region
            if self.dynamic_public_target_enabled and current_context is not None and current_context.executor_knows_target and (current_context.executor_navigation_target is not None):
                query_target = current_context.executor_navigation_target
                executor_public_target_shift = float(np.linalg.norm(np.asarray(query_target, dtype=np.float64) - np.asarray(executor_assignment.target_region, dtype=np.float64)))
                public_target_updated = bool(previous_context is not None and previous_context.executor_knows_target and (not current_context.mission_complete) and (executor_public_target_shift >= self.public_target_update_distance))
            executor_start_endpoint_current = self._endpoint_snapshot_present(current, executor.position, executor.role)
            executor_goal_endpoint_current = self._endpoint_snapshot_present(current, query_target, executor.role)
            executor_query = None
            if not executor_assignment_reachable:
                executor_invalid = True
                executor_query_reachable = False
                executor_invalid_reason = 'ASSIGNMENT_UNREACHABLE'
            elif self.defer_stale_endpoint_invalid and (not (executor_start_endpoint_current and executor_goal_endpoint_current)):
                executor_invalid = False
                executor_validity_evaluated = False
                executor_validity_deferred = True
                executor_query_reachable = True
                executor_invalid_reason = 'STALE_ENDPOINT_SNAPSHOT_DEFERRED'
            else:
                executor_query = service.query(executor.position, query_target, executor)
                executor_query_reachable = bool(executor_query.reachable)
                executor_query_failure_reason = str(executor_query.failure_reason or '')
                executor_current_planning_cost = float(executor_query.planning_cost)
                if not executor_query.reachable:
                    executor_invalid = True
                    executor_invalid_reason = 'QUERY_UNREACHABLE'
                elif public_target_updated:
                    executor_invalid = False
                    executor_invalid_reason = 'VALID'
                else:
                    executor_relative_cost_change = self._relative_cost_change(float(executor_query.planning_cost), float(executor_assignment.planning_cost))
                    executor_invalid = bool(executor_relative_cost_change > self.executor_cost_increase_threshold)
                    executor_invalid_reason = 'PLANNING_COST_INCREASE' if executor_invalid else 'VALID'
            stale = []
            for item in assignment.search_assignments:
                agent = current.agents[item.agent_id]
                distance = float(np.linalg.norm(np.asarray(agent.position, dtype=np.float64) - np.asarray(item.waypoint, dtype=np.float64)))
                start_current = self._endpoint_snapshot_present(current, agent.position, agent.role)
                query_unreachable = False
                if not self.defer_stale_endpoint_invalid or start_current:
                    query = service.query(agent.position, item.waypoint, agent)
                    query_unreachable = not query.reachable
                if query_unreachable or distance <= self.waypoint_stale_distance:
                    stale.append(item.agent_id)
            assignment_waypoints = tuple(sorted((*((item.agent_id, item.waypoint) for item in assignment.search_assignments), (executor_assignment.executor_id, executor_assignment.target_region))))
        else:
            target_region = tuple((float(value) for value in current.grid.cell_centers[current.target_belief.peak_index]))
            executor_query = service.query(executor.position, target_region, executor)
            executor_invalid = not executor_query.reachable
            executor_query_reachable = bool(executor_query.reachable)
            executor_query_failure_reason = str(executor_query.failure_reason or '')
            executor_current_planning_cost = float(executor_query.planning_cost)
            executor_invalid_reason = 'QUERY_UNREACHABLE' if executor_invalid else 'VALID'
            stale = []
            for agent_id in current.searcher_ids:
                agent = current.agents[agent_id]
                if agent.current_navigation_target is None:
                    stale.append(agent_id)
                    continue
                distance = float(np.linalg.norm(np.asarray(agent.position) - np.asarray(agent.current_navigation_target)))
                query = service.query(agent.position, agent.current_navigation_target, agent)
                if not query.reachable or distance <= self.waypoint_stale_distance:
                    stale.append(agent_id)
        detected = []
        if self.enable_belief_trigger and belief_score > self.belief_shift_threshold:
            detected.append(BSEREvent.BELIEF_SHIFT)
        if self.enable_obstacle_trigger and new_mass > self.obstacle_mass_threshold:
            detected.append(BSEREvent.OBSTACLE_DISCOVERED)
        if self.enable_target_trigger and (not previous.target_found) and current.target_found:
            detected.append(BSEREvent.TARGET_FOUND)
        target_received = bool(self.mechanism_version in {'phase1b1_corrected', 'phase1b2_corrected'} and previous_context is not None and (current_context is not None) and (not previous_context.executor_knows_target) and current_context.executor_knows_target)
        if target_received:
            detected.append(BSEREvent.EXECUTOR_TARGET_RECEIVED)
        if public_target_updated:
            detected.append(BSEREvent.EXECUTOR_PUBLIC_TARGET_UPDATED)
        if previous.target_found and (not current.target_found):
            detected.append(BSEREvent.TARGET_LOST)
        if executor_invalid:
            detected.append(BSEREvent.EXECUTOR_INVALID)
        if stale:
            detected.append(BSEREvent.WAYPOINT_STALE)
        if self.periodic_refresh_steps > 0 and current.step > previous.step and (current.step % self.periodic_refresh_steps == 0):
            detected.append(BSEREvent.PERIODIC_REFRESH)
        ordered = tuple((event for event in BSEREvent if event in detected))
        return EventDetection(events=ordered, previous_entropy=previous_entropy, current_entropy=current_entropy, belief_distance=belief_distance, belief_shift_score=belief_score, new_obstacle_cells=int(np.count_nonzero(new_cells)), new_obstacle_probability_mass=new_mass, risk_change=risk_change, executor_reachable=bool(executor_assignment_reachable if executor_validity_deferred else executor_query_reachable), stale_searcher_ids=tuple(stale), executor_target_received=target_received, executor_planning_cost_relative_change=executor_relative_cost_change, assignment_waypoints=assignment_waypoints, executor_validity_evaluated=executor_validity_evaluated, executor_validity_deferred=executor_validity_deferred, executor_invalid_reason=executor_invalid_reason, executor_query_failure_reason=executor_query_failure_reason, executor_assignment_reachable=executor_assignment_reachable, executor_query_reachable=executor_query_reachable, executor_start_endpoint_current=executor_start_endpoint_current, executor_goal_endpoint_current=executor_goal_endpoint_current, executor_installed_planning_cost=executor_installed_planning_cost, executor_current_planning_cost=executor_current_planning_cost, executor_public_target_shift=executor_public_target_shift)
