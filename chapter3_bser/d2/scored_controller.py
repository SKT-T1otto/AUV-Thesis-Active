from chapter3_bser.events.trigger_policy import TriggerDecision
from chapter3_bser.safety.failure_policy import SearchFailurePolicy
from chapter3_bser.online.controller import OnlineBSERController
from chapter3_bser.online.waypoint_manager import WaypointManager

class EffectPolicy(SearchFailurePolicy):

    def decide(self, events, new_objective, old_objective, step):
        events = tuple(events)
        if not self.search_enabled:
            return super().decide(events, new_objective, old_objective, step)
        event = self.primary_event(events)
        remaining = self.event_cooldown_remaining(event, step)
        gain = float(new_objective - old_objective)
        if event is None:
            return TriggerDecision(False, 'NO_REPLAN_EVENT', gain, 0)
        if remaining:
            return TriggerDecision(False, 'REJECT_EVENT_COOLDOWN', gain, remaining)
        return TriggerDecision(True, 'D_ACCEPT_ELIGIBLE_PROPOSAL', gain, 0)

class ScoredWaypoints(WaypointManager):

    def __init__(self, controller, previous):
        super().__init__(previous.tolerance, agent_cooldown_steps=previous.agent_cooldown_steps)
        self.controller = controller

    def stabilize(self, previous, proposed, **kwargs):
        result = super().stabilize(previous, proposed, **kwargs)
        state = self.controller.scoring_state
        return result if state is None else self.controller.allocator.rescore(state, result, finalize=True)

class EffectController(OnlineBSERController):

    def __init__(self, config, allocator):
        super().__init__(config, allocator=allocator)
        self.scoring_state = None
        self.waypoints = ScoredWaypoints(self, self.waypoints)

    def step(self, state, mission_context=None):
        if state.target_found:
            return super().step(state, mission_context)
        self.scoring_state = state
        old_value = self.current_allocation.objective_value
        self.current_allocation = self.allocator.rescore(state, self.current_allocation)
        old_current = self.current_allocation.objective_value
        try:
            result = super().step(state, mission_context)
            diagnostics = result.diagnostics
            if diagnostics and diagnostics.optimizer_invoked:
                self.allocator.audit.append(dict(kind='decision', step=int(state.step), previous_stored_objective=old_value, old_current_objective=old_current, objective=result.allocation.objective_value, installed=result.replanned, reason=result.decision_reason, selected_ids=[c.candidate_id for c in result.allocation.search_assignments], standby=list(result.allocation.executor_assignment.target_region)))
            return result
        finally:
            self.scoring_state = None
