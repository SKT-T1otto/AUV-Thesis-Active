from __future__ import annotations
from typing import Iterable
from typing import Mapping
from chapter3_bser.controllers.path_tracker import PathTracker
from chapter3_bser.integration.control_context import AgentAssignmentContextV1
from chapter3_bser.integration.control_context import BSERControlContextV1
from chapter3_bser.integration.control_context import ExecutorAssignmentContextV1
from chapter3_bser.integration.control_context import Vector3
from chapter3_bser.online.mission_context import OnlineMissionContext
from chapter3_bser.online.types import OnlineAllocation
from core.mapping.planning_state import PlanningStateView

def _vector3(value: Iterable[float]) -> Vector3:
    vector = tuple((float(item) for item in value))
    if len(vector) != 3:
        raise ValueError('Phase 1C guidance coordinates must be three-dimensional')
    return vector

def _mission_phase(context: OnlineMissionContext) -> str:
    if context.mission_complete:
        return 'MISSION_COMPLETE'
    if context.target_found and context.executor_knows_target:
        return 'EXECUTION'
    if context.target_found:
        return 'HANDOFF_PENDING'
    return 'SEARCH'

class RMADDPGGuidanceBridge:
    """Stateful path-tracking bridge with no action-generation capability."""
    SCHEMA_VERSION = 'bser.control_context.v1'

    def __init__(self, path_tracker: PathTracker | None=None) -> None:
        self.path_tracker = path_tracker or PathTracker()

    def compile_guidance(self, allocation: OnlineAllocation, planning_state: PlanningStateView, mission_context: OnlineMissionContext, *, decision_reason: str | None=None) -> BSERControlContextV1:
        """Compile BSER output and public state into immutable guidance.

        This method advances only the bridge-owned ``PathTracker``.  It never
        mutates the allocation, planning state, mission context, or environment.
        """
        if int(planning_state.step) != int(mission_context.step):
            raise ValueError('planning state and mission context steps differ')
        if int(allocation.executor_assignment.executor_id) != int(planning_state.executor_id):
            raise ValueError('allocation executor does not match planning state')
        agents_by_id = {int(agent.agent_id): agent for agent in planning_state.agents}
        if len(agents_by_id) != len(planning_state.agents):
            raise ValueError('planning state contains duplicate agent ids')
        search_by_id = {int(item.agent_id): item for item in allocation.search_assignments}
        if len(search_by_id) != len(allocation.search_assignments):
            raise ValueError('allocation contains duplicate search assignments')
        search_finished = {int(agent_id): bool(finished) for (agent_id, finished) in zip(planning_state.searcher_ids, mission_context.searcher_finished_flags)}
        execution_request = bool(mission_context.target_found and mission_context.executor_knows_target and (not mission_context.mission_complete))
        active_ids = tuple(sorted(agents_by_id))
        self.path_tracker.prune(active_ids)
        compiled: list[AgentAssignmentContextV1] = []
        for agent_id in active_ids:
            agent = agents_by_id[agent_id]
            position = _vector3(agent.position)
            if agent_id == int(planning_state.executor_id):
                executor = allocation.executor_assignment
                reachable = bool(executor.reachable)
                hold = bool(not reachable or mission_context.mission_complete)
                final = _vector3(executor.target_region)
                path = tuple((_vector3(point) for point in executor.path))
                tracking = position if hold else self.path_tracker.tracking_target(agent_id, position, path, final)
                compiled.append(AgentAssignmentContextV1(agent_id=agent_id, role=str(agent.role), assignment_kind='executor_execution' if execution_request else 'executor_standby', assignment_id=str(executor.source), final_waypoint=final, planned_path=path, tracking_waypoint=_vector3(tracking), hold_position=position, hold_state=hold, reachable=reachable, execution_request=execution_request))
                continue
            search = search_by_id.get(agent_id)
            finished = bool(search_finished.get(agent_id, False))
            reachable = bool(search is not None and search.failure_reason is None)
            hold = bool(search is None or finished or (not reachable))
            final = position if search is None else _vector3(search.waypoint)
            path = () if search is None else tuple((_vector3(point) for point in search.path))
            tracking = position if hold else self.path_tracker.tracking_target(agent_id, position, path, final)
            compiled.append(AgentAssignmentContextV1(agent_id=agent_id, role=str(agent.role), assignment_kind='hold' if hold else 'search', assignment_id='UNASSIGNED' if search is None else str(search.candidate_id), final_waypoint=final, planned_path=path, tracking_waypoint=_vector3(tracking), hold_position=position, hold_state=hold, reachable=reachable, execution_request=False))
        executor_agent = next((item for item in compiled if item.agent_id == int(planning_state.executor_id)))
        executor = allocation.executor_assignment
        allocation_hash = str(allocation.allocation_sha256)
        return BSERControlContextV1(schema_version=self.SCHEMA_VERSION, allocation_version=f'v1:{allocation_hash}', allocation_hash=allocation_hash, step=int(planning_state.step), mission_phase=_mission_phase(mission_context), agent_assignments=tuple(compiled), executor_assignment=ExecutorAssignmentContextV1(executor_id=int(executor.executor_id), source=str(executor.source), target_region=_vector3(executor.target_region), planned_path=tuple((_vector3(point) for point in executor.path)), tracking_waypoint=executor_agent.tracking_waypoint, hold_position=executor_agent.hold_position, hold_state=executor_agent.hold_state, reachable=executor_agent.reachable, execution_request=executor_agent.execution_request), decision_reason=str(allocation.trigger_reason if decision_reason is None else decision_reason))

def get_tracking_targets(context: BSERControlContextV1) -> Mapping[int, Vector3]:
    """Return each agent's waypoint or hold position; never an action."""
    return {item.agent_id: item.hold_position if item.hold_state else item.tracking_waypoint for item in context.agent_assignments}
