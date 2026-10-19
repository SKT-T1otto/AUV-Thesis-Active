from __future__ import annotations
from typing import Mapping
import numpy as np
from chapter3_bser.events.event_detector import EventDetector
from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.hysteresis.policy import ReplanningPolicy
from chapter3_bser.online.allocator import BSEROnlineAllocator
from chapter3_bser.online.config import execution_runtime_config
from chapter3_bser.online.config import load_phase1b_config
from chapter3_bser.online.mission_context import OnlineMissionContext
from chapter3_bser.online.route_impact import RouteImpactEvaluator
from chapter3_bser.online.route_impact import RouteImpactResult
from chapter3_bser.online.state_cache import OnlineStateCache
from chapter3_bser.online.types import BSERActionAssignment
from chapter3_bser.online.types import InitialBSERAllocation
from chapter3_bser.online.types import OnlineAllocation
from chapter3_bser.online.types import OnlineStepDiagnostics
from chapter3_bser.online.waypoint_manager import WaypointManager
from core.mapping.planning_state import PlanningStateView
from core.mapping.travel_cost_service import TravelCostService

class OnlineBSERController:
    """Pure high-level controller; it never emits velocity or acceleration."""

    def __init__(self, config: Mapping | None=None, allocator: BSEROnlineAllocator | None=None):
        self.config = dict(config or load_phase1b_config())
        self.mechanism_version = str(self.config.get('mechanism_version', 'phase1b_v1'))
        self.execution_runtime = execution_runtime_config(self.config)
        self.detector = EventDetector(self.config)
        self.policy = ReplanningPolicy(self.config)
        self.allocator = allocator or BSEROnlineAllocator()
        hysteresis = self.config['hysteresis']
        tolerance = 1e-09 if self.mechanism_version == 'phase1b_v1' else float(hysteresis['minimum_waypoint_switch_distance'])
        agent_cooldown = 0 if self.mechanism_version == 'phase1b_v1' else int(hysteresis['waypoint_stale_cooldown_steps'])
        self.waypoints = WaypointManager(tolerance, agent_cooldown_steps=agent_cooldown)
        route_config = self.config.get('route_impact', {})
        self.route_impact = RouteImpactEvaluator(planning_cost_threshold=float(route_config.get('planning_cost_relative_threshold', 0.15)), corridor_mass_threshold=float(route_config.get('corridor_probability_mass_threshold', 0.2)))
        self.cache = OnlineStateCache()
        self.current_allocation: OnlineAllocation | None = None
        self.current_context: OnlineMissionContext | None = None
        self.execution_target: tuple[float, float, float] | None = None
        self.execution_target_source: str | None = None
        self.last_execution_target_update_step: int | None = None
        self.public_target_update_count = 0
        self.replan_count = 0
        self.replan_steps: list[int] = []

    @property
    def current_allocation(self) -> OnlineAllocation | None:
        """Compatibility view backed by the canonical WaypointManager state."""
        return self.waypoints.current_assignment

    @current_allocation.setter
    def current_allocation(self, allocation: OnlineAllocation | None) -> None:
        self.waypoints.commit(allocation)

    def initialize(self, state: PlanningStateView, mission_context: OnlineMissionContext | None=None) -> InitialBSERAllocation:
        if self.mechanism_version in {'phase1b1_corrected', 'phase1b2_corrected'} and mission_context is None:
            raise ValueError('corrected mechanism requires public mission context')
        allocation = self.allocator.allocate(state, trigger_reason='INITIALIZE')
        updates = self.waypoints.updates(None, allocation, reason='INITIALIZE', step=state.step)
        self.current_allocation = allocation
        self.waypoints.commit(allocation)
        self.current_context = mission_context or OnlineMissionContext.from_planning_view(state)
        self.execution_target = None
        self.execution_target_source = None
        self.last_execution_target_update_step = None
        self.public_target_update_count = 0
        if self.mechanism_version == 'phase1b2_corrected' and self.current_context.executor_knows_target and (self.current_context.executor_navigation_target is not None):
            self.execution_target = tuple((float(value) for value in self.current_context.executor_navigation_target))
            self.execution_target_source = 'PUBLIC_HANDOFF'
            self.last_execution_target_update_step = int(state.step)
        self.cache.initialize(state)
        self.policy.mark_replan(state.step)
        self.replan_count = 1
        self.replan_steps = [int(state.step)]
        return InitialBSERAllocation(allocation, updates)

    @staticmethod
    def _waypoints(allocation: OnlineAllocation):
        return tuple(((item.agent_id, item.waypoint) for item in allocation.search_assignments))

    def _diagnostics(self, *, state: PlanningStateView, events, optimizer_invoked: bool, scope: str, old: OnlineAllocation, proposed: OnlineAllocation, accepted: bool, accept_reason: str='', reject_reason: str='', affected=(), route_impacted: bool=False, updates=()) -> OnlineStepDiagnostics:
        distances = []
        for update in updates:
            distance = 0.0 if update.old_waypoint is None else float(np.linalg.norm(np.asarray(update.new_waypoint) - np.asarray(update.old_waypoint)))
            distances.append((update.agent_id, distance))
        return OnlineStepDiagnostics(step=int(state.step), mechanism_version=self.mechanism_version, detected_events=tuple((event.value for event in events)), optimizer_invoked=bool(optimizer_invoked), allocation_scope=str(scope), old_objective=float(old.objective_value), proposed_objective=float(proposed.objective_value), objective_gain=float(proposed.objective_value - old.objective_value), accepted=bool(accepted), accept_reason=str(accept_reason), reject_reason=str(reject_reason), affected_agent_ids=tuple(sorted((int(value) for value in affected))), obstacle_route_impacted=bool(route_impacted), old_waypoints=self._waypoints(old), proposed_waypoints=self._waypoints(proposed), switch_distance_by_agent=tuple(distances), executor_target_source=str(proposed.executor_assignment.source))

    def step(self, state: PlanningStateView, mission_context: OnlineMissionContext | None=None) -> BSERActionAssignment:
        if self.waypoints.current_assignment is None or self.cache.current is None:
            raise RuntimeError('initialize must be called before step')
        if self.mechanism_version == 'phase1b_v1':
            return self._step_v1(state)
        if mission_context is None:
            raise ValueError('corrected mechanism requires public mission context')
        return self._step_corrected(state, mission_context)

    def _step_v1(self, state: PlanningStateView) -> BSERActionAssignment:
        previous_state = self.cache.current
        detection = self.detector.detect(previous_state, state)
        self.cache.update(state)
        events = detection.events
        if not events:
            return BSERActionAssignment(state.step, False, events, self.current_allocation, (), 'no_event', detection)
        critical = any((event in self.policy.CRITICAL_EVENTS for event in events))
        if not critical and self.policy.enabled and (not self.policy.cooldown.ready(state.step)):
            decision = self.policy.decide(events, self.current_allocation.objective_value, self.current_allocation.objective_value, state.step)
            return BSERActionAssignment(state.step, False, events, self.current_allocation, (), decision.reason, detection)
        if BSEREvent.TARGET_FOUND in events:
            proposed = self.allocator.reassign_after_target_found(state)
        else:
            proposed = self.allocator.allocate(state, trigger_reason='+'.join((event.value for event in events)))
        decision = self.policy.decide(events, proposed.objective_value, self.current_allocation.objective_value, state.step)
        if not decision.should_replan:
            return BSERActionAssignment(state.step, False, events, self.current_allocation, (), decision.reason, detection)
        previous = self.current_allocation
        self.current_allocation = proposed
        self.waypoints.commit(proposed)
        updates = self.waypoints.updates(previous, proposed, reason=decision.reason, step=state.step)
        self.policy.mark_replan(state.step)
        self.replan_count += 1
        self.replan_steps.append(int(state.step))
        return BSERActionAssignment(state.step, True, events, proposed, updates, decision.reason, detection)

    def _step_corrected(self, state: PlanningStateView, mission_context: OnlineMissionContext) -> BSERActionAssignment:
        previous_state = self.cache.current
        previous_context = self.current_context
        old = self.waypoints.current_assignment
        if old is None:
            raise RuntimeError('canonical waypoint assignment is unavailable')
        detection = self.detector.detect(previous_state, state, previous_context, mission_context, assignment=old if self.mechanism_version == 'phase1b2_corrected' else None)
        self.cache.update(state)
        self.current_context = mission_context
        if self.mechanism_version == 'phase1b2_corrected' and self.execution_target is None and mission_context.executor_knows_target and (mission_context.executor_navigation_target is not None):
            self.execution_target = tuple((float(value) for value in mission_context.executor_navigation_target))
            self.execution_target_source = 'PUBLIC_HANDOFF'
            self.last_execution_target_update_step = int(state.step)
        if self.execution_runtime['dynamic_public_target_enabled'] and (mission_context.mission_complete or BSEREvent.TARGET_LOST in detection.events):
            self.execution_target = None
            self.execution_target_source = None
            self.last_execution_target_update_step = None
        execution_phase = bool(self.mechanism_version == 'phase1b2_corrected' and self.execution_target is not None and mission_context.executor_knows_target and (not mission_context.mission_complete))
        detected = detection.events
        actionable = list(detected)
        corrected_stale = tuple(detection.stale_searcher_ids)
        if self.mechanism_version == 'phase1b1_corrected' and BSEREvent.WAYPOINT_STALE in actionable:
            service = TravelCostService(state)
            threshold = float(self.config['events']['waypoint_stale_distance'])
            stale = []
            for assignment in old.search_assignments:
                agent = state.agents[assignment.agent_id]
                distance = float(np.linalg.norm(np.asarray(agent.position) - np.asarray(assignment.waypoint)))
                query = service.query(agent.position, assignment.waypoint, agent)
                if distance <= threshold or not query.reachable:
                    stale.append(assignment.agent_id)
            corrected_stale = tuple(stale)
            if not corrected_stale:
                actionable.remove(BSEREvent.WAYPOINT_STALE)
        impact: RouteImpactResult | None = None
        if BSEREvent.OBSTACLE_DISCOVERED in actionable:
            impact = self.route_impact.evaluate(previous_state, state, old)
            if not impact.route_impacted:
                actionable.remove(BSEREvent.OBSTACLE_DISCOVERED)
        if execution_phase:
            if BSEREvent.OBSTACLE_DISCOVERED in actionable and (not bool(impact and impact.executor_route_impacted)):
                actionable.remove(BSEREvent.OBSTACLE_DISCOVERED)
            allowed = {BSEREvent.EXECUTOR_TARGET_RECEIVED, BSEREvent.EXECUTOR_PUBLIC_TARGET_UPDATED, BSEREvent.EXECUTOR_INVALID, BSEREvent.OBSTACLE_DISCOVERED}
            actionable = [event for event in actionable if event in allowed]
            corrected_stale = ()
        if BSEREvent.EXECUTOR_PUBLIC_TARGET_UPDATED in actionable:
            update_target = mission_context.executor_navigation_target
            last_step = self.last_execution_target_update_step
            update_interval = int(state.step) - int(last_step) if last_step is not None else self.execution_runtime['public_target_update_min_steps']
            update_shift = 0.0 if update_target is None or self.execution_target is None else float(np.linalg.norm(np.asarray(update_target, dtype=np.float64) - np.asarray(self.execution_target, dtype=np.float64)))
            if not self.execution_runtime['dynamic_public_target_enabled'] or update_target is None or update_shift < float(self.execution_runtime['public_target_update_distance']) or (update_interval < int(self.execution_runtime['public_target_update_min_steps'])):
                actionable.remove(BSEREvent.EXECUTOR_PUBLIC_TARGET_UPDATED)
        if self.mechanism_version == 'phase1b1_corrected' and BSEREvent.EXECUTOR_INVALID in actionable:
            obstacle_invalidated = bool(impact is not None and impact.executor_route_impacted)
            if old.executor_assignment.reachable and (not obstacle_invalidated):
                actionable.remove(BSEREvent.EXECUTOR_INVALID)
            else:
                preserved = self.allocator.execution.preserve_current(state, old.executor_assignment)
                if preserved.reachable:
                    actionable.remove(BSEREvent.EXECUTOR_INVALID)
        waiting = BSEREvent.TARGET_FOUND in detected and (not mission_context.executor_knows_target)
        if waiting and BSEREvent.TARGET_FOUND in actionable:
            actionable.remove(BSEREvent.TARGET_FOUND)
        primary = self.policy.primary_event(actionable)
        if primary is None:
            if waiting:
                reason = 'WAITING_FOR_PUBLIC_HANDOFF'
            elif execution_phase and detected:
                reason = 'EXECUTION_TARGET_LOCKED'
            elif impact is not None:
                reason = 'OBSTACLE_OFF_ROUTE'
            else:
                reason = 'NO_REPLAN_EVENT'
            diagnostics = self._diagnostics(state=state, events=detected, optimizer_invoked=False, scope='none', old=old, proposed=old, accepted=False, reject_reason=reason, affected=() if impact is None else impact.affected_agent_ids, route_impacted=False if impact is None else impact.route_impacted)
            return BSERActionAssignment(state.step, False, detected, old, (), reason, detection, diagnostics)
        remaining = self.policy.event_cooldown_remaining(primary, state.step)
        if remaining and primary != BSEREvent.EXECUTOR_TARGET_RECEIVED:
            reason = 'REJECT_EVENT_COOLDOWN'
            diagnostics = self._diagnostics(state=state, events=detected, optimizer_invoked=False, scope='none', old=old, proposed=old, accepted=False, reject_reason=reason, affected=() if impact is None else impact.affected_agent_ids, route_impacted=False if impact is None else impact.route_impacted)
            return BSERActionAssignment(state.step, False, detected, old, (), reason, detection, diagnostics)
        optimizer_invoked = False
        atomic_ok = True
        atomic_reason = ''
        if primary == BSEREvent.EXECUTOR_TARGET_RECEIVED:
            scope = 'executor_public_target'
            affected = (state.executor_id,)
            if self.mechanism_version == 'phase1b2_corrected':
                (proposed, atomic_ok, atomic_reason) = self.allocator.replan_executor_to_public_target(state, old, self.execution_target, trigger_reason='EXECUTOR_TARGET_RECEIVED')
            else:
                proposed = self.allocator.reassign_after_target_received(state, old, mission_context.executor_navigation_target)
        elif primary == BSEREvent.EXECUTOR_PUBLIC_TARGET_UPDATED:
            scope = 'executor_public_target'
            affected = (state.executor_id,)
            (proposed, atomic_ok, atomic_reason) = self.allocator.replan_executor_to_public_target(state, old, mission_context.executor_navigation_target, trigger_reason='EXECUTOR_PUBLIC_TARGET_UPDATED')
        elif primary == BSEREvent.EXECUTOR_INVALID:
            optimizer_invoked = self.mechanism_version == 'phase1b2_corrected'
            affected = (state.executor_id,)
            if execution_phase:
                scope = 'executor_public_target'
                (proposed, atomic_ok, atomic_reason) = self.allocator.replan_executor_to_public_target(state, old, self.execution_target, trigger_reason='EXECUTOR_INVALID')
            elif self.mechanism_version == 'phase1b2_corrected':
                scope = 'executor_only'
                (proposed, atomic_ok, atomic_reason) = self.allocator.allocate_partial(state, old, affected_searcher_ids=(), executor_affected=True, trigger_reason='EXECUTOR_INVALID')
            else:
                scope = 'executor_only'
                proposed = self.allocator.reassign_invalid_executor(state, old)
        elif primary == BSEREvent.OBSTACLE_DISCOVERED and impact is not None:
            optimizer_invoked = True
            if execution_phase:
                affected = (state.executor_id,)
                scope = 'executor_public_target'
                (proposed, atomic_ok, atomic_reason) = self.allocator.replan_executor_to_public_target(state, old, self.execution_target, trigger_reason='OBSTACLE_ROUTE_IMPACT')
            else:
                affected = impact.affected_agent_ids
                scope = 'affected_agents'
                (proposed, atomic_ok, atomic_reason) = self.allocator.allocate_partial(state, old, affected_searcher_ids=impact.affected_searcher_ids, executor_affected=impact.executor_route_impacted, trigger_reason='OBSTACLE_ROUTE_IMPACT')
        elif primary == BSEREvent.WAYPOINT_STALE:
            optimizer_invoked = True
            affected = corrected_stale
            scope = 'stale_searchers'
            (proposed, atomic_ok, atomic_reason) = self.allocator.allocate_partial(state, old, affected_searcher_ids=affected, executor_affected=False, trigger_reason='WAYPOINT_STALE')
        else:
            optimizer_invoked = True
            affected = state.searcher_ids
            scope = 'all_searchers_and_executor'
            proposed = self.allocator.allocate(state, trigger_reason=primary.value)
        if not atomic_ok:
            self.policy.mark_attempt(state.step, primary)
            diagnostics = self._diagnostics(state=state, events=detected, optimizer_invoked=optimizer_invoked, scope=scope, old=old, proposed=old, accepted=False, reject_reason=atomic_reason, affected=affected, route_impacted=bool(impact and impact.route_impacted))
            return BSERActionAssignment(state.step, False, detected, old, (), atomic_reason, detection, diagnostics)
        decision = self.policy.decide(actionable, proposed.objective_value, old.objective_value, state.step)
        if not decision.should_replan:
            if optimizer_invoked:
                self.policy.mark_attempt(state.step, primary)
            diagnostics = self._diagnostics(state=state, events=detected, optimizer_invoked=optimizer_invoked, scope=scope, old=old, proposed=proposed, accepted=False, reject_reason=decision.reason, affected=affected, route_impacted=bool(impact and impact.route_impacted))
            return BSERActionAssignment(state.step, False, detected, old, (), decision.reason, detection, diagnostics)
        proposed = self.waypoints.stabilize(old, proposed, affected_agent_ids=affected, step=state.step)
        updates = self.waypoints.updates(old, proposed, reason=decision.reason, step=state.step)
        executor_changed = proposed.executor_assignment.target_region != old.executor_assignment.target_region or proposed.executor_assignment.source != old.executor_assignment.source or proposed.executor_assignment.path != old.executor_assignment.path or (proposed.executor_assignment.path_cell_indices != old.executor_assignment.path_cell_indices) or (proposed.executor_assignment.reachable != old.executor_assignment.reachable) or (not np.isclose(proposed.executor_assignment.planning_cost, old.executor_assignment.planning_cost, equal_nan=True))
        if not updates and (not executor_changed) and (primary not in {BSEREvent.EXECUTOR_TARGET_RECEIVED, BSEREvent.EXECUTOR_PUBLIC_TARGET_UPDATED}):
            self.policy.mark_attempt(state.step, primary)
            reason = 'REJECT_NO_ASSIGNMENT_CHANGE'
            diagnostics = self._diagnostics(state=state, events=detected, optimizer_invoked=optimizer_invoked, scope=scope, old=old, proposed=proposed, accepted=False, reject_reason=reason, affected=affected, route_impacted=bool(impact and impact.route_impacted))
            return BSERActionAssignment(state.step, False, detected, old, (), reason, detection, diagnostics)
        self.current_allocation = proposed
        self.waypoints.commit(proposed)
        self.policy.mark_replan(state.step, primary)
        self.replan_count += 1
        self.replan_steps.append(int(state.step))
        if primary == BSEREvent.EXECUTOR_PUBLIC_TARGET_UPDATED:
            self.execution_target = tuple((float(value) for value in mission_context.executor_navigation_target))
            self.execution_target_source = 'PUBLIC_HANDOFF'
            self.last_execution_target_update_step = int(state.step)
            self.public_target_update_count += 1
        diagnostics = self._diagnostics(state=state, events=detected, optimizer_invoked=optimizer_invoked, scope=scope, old=old, proposed=proposed, accepted=True, accept_reason=decision.reason, affected=affected, route_impacted=bool(impact and impact.route_impacted), updates=updates)
        return BSERActionAssignment(state.step, True, detected, proposed, updates, decision.reason, detection, diagnostics)
