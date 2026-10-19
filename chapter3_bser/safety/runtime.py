from collections import Counter
import numpy as np
from chapter3_bser.events.event_types import BSEREvent
from chapter3_bser.online.types import BSERActionAssignment
from chapter3_bser.safety.failure_policy import SearchFailurePolicy
from chapter3_bser.safety.planning_refresh import PlanningViews
from chapter3_bser.safety.planning_refresh import start_status
from chapter3_bser.safety.planning_refresh import topology_key
from chapter3_bser.safety.public_geometry import PublicGeometry
VARIANTS = {'V0': dict(planning_state_consistency=False, failure_policy=False, path_safety=False), 'V1': dict(planning_state_consistency=True, failure_policy=False, path_safety=False), 'V2': dict(planning_state_consistency=True, failure_policy=True, path_safety=False), 'V3': dict(planning_state_consistency=False, failure_policy=False, path_safety=True), 'V4': dict(planning_state_consistency=True, failure_policy=True, path_safety=True), 'V5': dict(planning_state_consistency=True, failure_policy=False, path_safety=True)}

class SearchController:
    """Compose the original controller; no replacement event/physics engine."""

    def __init__(self, inner, owner):
        self.inner = inner
        self.owner = owner
        if owner.safe_options['failure_policy']:
            self.inner.policy = SearchFailurePolicy(inner.config)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def initialize(self, state, context=None):
        return self.inner.initialize(state, context)

    def _geometry_pending(self, state):
        if not self.owner.safe_options['path_safety']:
            return set()
        geometry = PublicGeometry(self.owner.planning_views.live(), self.owner.planning_views.clearance)
        pending = set(self.owner.safe_pending)
        if hasattr(self.owner, 'bridge'):
            for item in self.inner.current_allocation.search_assignments:
                remaining = self.owner.bridge.path_tracker.snapshot(item.agent_id).remaining_path_points
                if not geometry.path_free((state.agents[item.agent_id].position, *remaining)):
                    pending.add(item.agent_id)
        return pending

    def step(self, state, mission_context=None):
        (owner, inner) = (self.owner, self.inner)
        if state.target_found:
            owner.safe_pending.clear()
            if isinstance(inner.policy, SearchFailurePolicy):
                inner.policy.search_enabled = False
            return inner.step(state, mission_context)
        detection = inner.detector.detect(inner.cache.current, state, inner.current_context, mission_context, assignment=inner.current_allocation)
        stale = set(detection.stale_searcher_ids)
        geometry_pending = self._geometry_pending(state)
        pending = set(owner.safe_pending) | geometry_pending
        if owner.safe_options['failure_policy']:
            finished = set((i for (i, flag) in zip(state.searcher_ids, getattr(mission_context, 'searcher_finished_flags', ())) if flag))
            assigned = {item.agent_id for item in inner.current_allocation.search_assignments}
            pending.update(set(state.searcher_ids) - assigned - finished)
        primary = inner.policy.primary_event(detection.events)
        policy = inner.policy if isinstance(inner.policy, SearchFailurePolicy) else None
        ready = policy.ready(state, topology_key(state), full_refresh=owner.provider.last_snapshot_was_full_refresh) if policy else False
        if policy:
            policy.recovery_ready = ready and bool(pending or stale)
            policy.retry_remaining = 0 if ready or policy.last_failed_step is None else max(0, policy.retry_steps - (state.step - policy.last_failed_step))
        opportunity = primary is not None and (not inner.policy.event_cooldown_remaining(primary, state.step))
        if owner.safe_options['planning_state_consistency']:
            need_snapshot = bool(geometry_pending and (not policy or ready) or ((opportunity or (pending and ready)) and any((start_status(state, i) == 'stale_start_endpoint' for i in state.searcher_ids))))
            if need_snapshot:
                state = owner.planning_views.refresh()
        owner.state = state
        if policy:
            ready = policy.ready(state, topology_key(state), full_refresh=owner.provider.last_snapshot_was_full_refresh)
            policy.recovery_ready = ready and bool(pending or stale)
            policy.retry_remaining = 0 if ready or policy.last_failed_step is None else max(0, policy.retry_steps - (state.step - policy.last_failed_step))
        old = inner.current_allocation
        result = inner.step(state, mission_context)
        if policy:
            affected = (pending | stale) & set(state.searcher_ids)
            invoked = bool(result.diagnostics and result.diagnostics.optimizer_invoked)
            if result.replanned:
                policy.succeeded()
                installed = {item.agent_id for item in result.allocation.search_assignments}
                owner.safe_pending.difference_update(affected & installed)
                owner.safe_pending.update(affected - installed)
            elif affected and ready and (not invoked):
                (proposed, ok, reason) = inner.allocator.allocate_partial(state, inner.current_allocation, affected_searcher_ids=tuple(sorted(affected)), executor_affected=False, trigger_reason='SAFE_SEARCH_RECOVERY')
                if ok and proposed != old:
                    updates = inner.waypoints.updates(old, proposed, reason='SAFE_SEARCH_RECOVERY', step=state.step)
                    inner.current_allocation = proposed
                    inner.policy.mark_replan(state.step, BSEREvent.WAYPOINT_STALE)
                    inner.replan_count += 1
                    inner.replan_steps.append(int(state.step))
                    diag = inner._diagnostics(state=state, events=result.events, optimizer_invoked=True, scope='safe_search_recovery', old=old, proposed=proposed, accepted=True, accept_reason='SAFE_SEARCH_RECOVERY', affected=affected, route_impacted=bool(geometry_pending), updates=updates)
                    result = BSERActionAssignment(state.step, True, result.events, proposed, updates, 'SAFE_SEARCH_RECOVERY', result.event_detection, diag)
                    policy.succeeded()
                    owner.safe_pending.difference_update(affected)
                    owner.safe_counts['recovery_replans'] += 1
                else:
                    owner.safe_pending.update(affected)
                    policy.failed(state.step, topology_key(state))
                    owner.safe_counts['recovery_failure:' + reason] += 1
            elif affected and invoked:
                owner.safe_pending.update(affected)
                policy.failed(state.step, topology_key(state))
        return result

class SafetyMixin:

    def _setup_safe(self, scenario, variant):
        if variant not in VARIANTS or variant == 'V0':
            raise ValueError('enabled runtime requires V1..V5')
        self.safe_variant = variant
        self.safe_options = dict(VARIANTS[variant])
        self.safe_counts = Counter()
        self.safe_pending = set()
        clearance = float(scenario.get('planner_obstacle_clearance', 0.4))
        if not np.isfinite(clearance) or clearance < 0:
            raise ValueError('invalid public planner clearance')
        self.planning_views = PlanningViews(self, clearance)
