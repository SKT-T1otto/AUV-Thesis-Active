from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from typing import Tuple
from core.env.task_state import MissionTaskState
from core.env.task_state import SearchExecutionState
from core.mapping.planning_state import PlanningStateView
Vector3 = Tuple[float, float, float]

@dataclass(frozen=True)
class OnlineMissionContext:
    step: int
    target_found: bool
    finder_id: int
    executor_knows_target: bool
    handoff_step: Optional[int]
    mission_complete: bool
    executor_navigation_target: Optional[Vector3]
    target_known_by_agent: Tuple[bool, ...]
    searcher_finished_flags: Tuple[bool, ...]

    @classmethod
    def from_public_views(cls, task: MissionTaskState, search: SearchExecutionState, planning: PlanningStateView) -> 'OnlineMissionContext':
        if int(task.step) != int(planning.step):
            raise ValueError('public task and planning views must describe the same step')
        executor = planning.agents[planning.executor_id]
        return cls(step=int(task.step), target_found=bool(task.target_found), finder_id=int(task.finder_id), executor_knows_target=bool(task.executor_knows_target), handoff_step=None if task.handoff_step is None else int(task.handoff_step), mission_complete=bool(task.mission_complete), executor_navigation_target=None if executor.current_navigation_target is None else tuple((float(value) for value in executor.current_navigation_target)), target_known_by_agent=tuple((bool(value) for value in search.target_known_by_agent)), searcher_finished_flags=tuple((bool(search.agent_finished[index]) for index in planning.searcher_ids)))

    @classmethod
    def from_planning_view(cls, planning: PlanningStateView) -> 'OnlineMissionContext':
        """Compatibility context used only by the original mechanism."""
        executor = planning.agents[planning.executor_id]
        return cls(step=int(planning.step), target_found=bool(planning.target_found), finder_id=-1, executor_knows_target=False, handoff_step=None, mission_complete=bool(planning.mission_complete), executor_navigation_target=executor.current_navigation_target, target_known_by_agent=tuple((False for _ in planning.agents)), searcher_finished_flags=tuple((False for _ in planning.searcher_ids)))
