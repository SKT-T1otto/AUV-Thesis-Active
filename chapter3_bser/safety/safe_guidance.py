from dataclasses import replace
from chapter3_bser.controllers.path_tracker import PathTracker
from chapter3_bser.integration.rmaddpg_bridge import RMADDPGGuidanceBridge
from core.mapping.travel_cost_service import TravelCostService
from chapter3_bser.safety.public_geometry import PublicGeometry

def _signature(assignment):
    return (int(assignment.agent_id), str(assignment.candidate_id), tuple(assignment.waypoint), tuple((tuple(point) for point in assignment.path)), assignment.failure_reason)

class SafetyBridge(RMADDPGGuidanceBridge):
    """Retain semantic assignments while replacing or holding unsafe routes."""

    def __init__(self, owner, path_tracker=None):
        super().__init__(path_tracker)
        self.owner = owner
        self.last_safety_status = {}
        self._overrides = {}
        self._attempt_steps = {}
        self._counted = {}

    def _count_once(self, name, agent_id, step):
        key = (name, int(agent_id))
        if self._counted.get(key) != int(step):
            self.owner.safe_counts[name] += 1
            self._counted[key] = int(step)

    def _apply_overrides(self, allocation):
        compiled = []
        active = set()
        for assignment in allocation.search_assignments:
            agent_id = int(assignment.agent_id)
            active.add(agent_id)
            stored = self._overrides.get(agent_id)
            if stored is not None and stored[0] != _signature(assignment):
                self._overrides.pop(agent_id)
                stored = None
            if stored is not None:
                route = stored[1]
                assignment = replace(assignment, path=route.path, path_cell_indices=route.path_cell_indices, physical_travel_time=route.physical_travel_time, planning_cost=route.planning_cost, failure_reason=None)
            compiled.append(assignment)
        for agent_id in tuple(self._overrides):
            if agent_id not in active:
                self._overrides.pop(agent_id)
        return replace(allocation, search_assignments=tuple(compiled))

    def _remaining_safe(self, geometry, item, position):
        if not item.reachable or item.hold_state:
            return False
        remaining = self.path_tracker.snapshot(item.agent_id).remaining_path_points
        return geometry.segment_free(position, item.tracking_waypoint) and geometry.path_free((position, *remaining))

    def compile_guidance(self, allocation, planning_state, mission_context, *, decision_reason=None):
        if not self.owner.safe_options['path_safety'] or mission_context.target_found or planning_state.target_found or mission_context.mission_complete:
            self._overrides.clear()
            self.last_safety_status.clear()
            return super().compile_guidance(allocation, planning_state, mission_context, decision_reason=decision_reason)
        effective = self._apply_overrides(allocation)
        context = super().compile_guidance(effective, planning_state, mission_context, decision_reason=decision_reason)
        live = self.owner.planning_views.live()
        if int(live.step) != int(planning_state.step):
            raise ValueError('safety geometry must describe the current guidance step')
        geometry = PublicGeometry(live, self.owner.planning_views.clearance)
        live_agents = {int(agent.agent_id): agent for agent in live.agents}
        originals = {int(item.agent_id): item for item in allocation.search_assignments}
        finished = dict(zip(planning_state.searcher_ids, mission_context.searcher_finished_flags))
        step = int(planning_state.step)
        output = []
        service = None
        for item in context.agent_assignments:
            agent_id = int(item.agent_id)
            if agent_id == int(planning_state.executor_id):
                output.append(item)
                continue
            if finished.get(agent_id, False):
                self.last_safety_status[agent_id] = 'SEARCH_FINISHED'
                self.owner.safe_pending.discard(agent_id)
                output.append(item)
                continue
            original = originals.get(agent_id)
            if original is None:
                self.last_safety_status[agent_id] = 'UNASSIGNED'
                self.owner.safe_pending.add(agent_id)
                output.append(item)
                continue
            agent = live_agents[agent_id]
            position = tuple((float(value) for value in agent.position))
            self._count_once('path_safety_checks', agent_id, step)
            if self._remaining_safe(geometry, item, position):
                self.last_safety_status[agent_id] = 'SAFE'
                self.owner.safe_pending.discard(agent_id)
                output.append(item)
                continue
            self._count_once('unsafe_search_paths', agent_id, step)
            repaired = None
            if self._attempt_steps.get(agent_id) != step:
                self._attempt_steps[agent_id] = step
                self.owner.safe_counts['safety_route_queries'] += 1
                if service is None:
                    service = TravelCostService(self.owner.planning_views.search(self.owner.state))
                route = service.query(position, original.waypoint, agent)
                if route.reachable and geometry.path_free(route.path_points):
                    path = tuple((tuple((float(value) for value in point)) for point in route.path_points))
                    proposed = replace(original, path=path, path_cell_indices=tuple((int(value) for value in route.path_cell_indices)), physical_travel_time=float(route.physical_travel_time), planning_cost=float(route.planning_cost), failure_reason=None)
                    probe = PathTracker(threshold=self.path_tracker.threshold)
                    tracking = probe.tracking_target(agent_id, position, path, original.waypoint)
                    remaining = probe.snapshot(agent_id).remaining_path_points
                    if geometry.segment_free(position, tracking) and geometry.path_free((position, *remaining)):
                        self.path_tracker.reset(agent_id)
                        tracking = self.path_tracker.tracking_target(agent_id, position, path, original.waypoint)
                        self._overrides[agent_id] = (_signature(original), proposed)
                        repaired = replace(item, planned_path=path, tracking_waypoint=tracking, hold_position=position, hold_state=False, reachable=True, assignment_kind='search')
                        self.owner.safe_counts['safety_route_replacements'] += 1
                if repaired is None:
                    self.owner.safe_counts['safety_route_failures'] += 1
                    reason = route.failure_reason if not route.reachable else 'latest_public_geometry_rejected'
                    self.owner.safe_counts[f'safety_route_failure:{reason}'] += 1
            if repaired is not None:
                self.last_safety_status[agent_id] = 'SAFE_REROUTE'
                self.owner.safe_pending.discard(agent_id)
                output.append(repaired)
            else:
                self.last_safety_status[agent_id] = 'UNVERIFIED_BRAKING'
                self.owner.safe_pending.add(agent_id)
                self._count_once('safety_braking_agent_steps', agent_id, step)
                output.append(replace(item, planned_path=(), tracking_waypoint=position, hold_position=position, hold_state=True, reachable=False, assignment_kind='hold'))
        return replace(context, agent_assignments=tuple(output))
