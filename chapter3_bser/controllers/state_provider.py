from __future__ import annotations
from dataclasses import replace
from core.mapping.planning_state import PlanningStateView
from core.mapping.planning_state import extract_planning_state

class OnlinePlanningStateProvider:

    def __init__(self, env, refresh_interval: int=20, *, refresh_on_executor_handoff: bool=False, refresh_on_public_target_shift: bool=False, public_target_update_distance: float=0.75, public_target_update_min_steps: int=20):
        self.env = env
        self.refresh_interval = int(refresh_interval)
        self.refresh_on_executor_handoff = bool(refresh_on_executor_handoff)
        self.refresh_on_public_target_shift = bool(refresh_on_public_target_shift)
        self.public_target_update_distance = float(public_target_update_distance)
        self.public_target_update_min_steps = int(public_target_update_min_steps)
        self.cached: PlanningStateView | None = None
        self._cached_executor_knows_target = False
        self._last_full_refresh_step = 0
        self._last_full_refresh_executor_target = None
        self._last_snapshot_full_refresh = False
        self._last_refresh_reason = ''
        self._full_refresh_count = 0
        self._handoff_forced_refresh_count = 0
        self._target_shift_forced_refresh_count = 0

    @staticmethod
    def _executor_target(task, public):
        if not bool(task.executor_knows_target):
            return None
        roles = tuple((str(value).lower() for value in public.role_order))
        executor_id = next((index for (index, role) in enumerate(roles) if role.startswith('exec')), 3)
        return tuple((float(value) for value in public.navigation_targets[executor_id]))

    def _record_full_refresh(self, task, public, reason: str) -> None:
        self._cached_executor_knows_target = bool(task.executor_knows_target)
        self._last_full_refresh_step = int(task.step)
        self._last_full_refresh_executor_target = self._executor_target(task, public)
        self._last_snapshot_full_refresh = True
        self._last_refresh_reason = str(reason)
        self._full_refresh_count += 1

    @property
    def last_snapshot_was_full_refresh(self) -> bool:
        return bool(self._last_snapshot_full_refresh)

    @property
    def full_refresh_count(self) -> int:
        return int(self._full_refresh_count)

    @property
    def handoff_forced_refresh_count(self) -> int:
        return int(self._handoff_forced_refresh_count)

    @property
    def target_shift_forced_refresh_count(self) -> int:
        return int(self._target_shift_forced_refresh_count)

    def initialize(self) -> PlanningStateView:
        task = self.env.get_task_state()
        public = self.env.get_agent_state()
        self.cached = extract_planning_state(self.env)
        self._record_full_refresh(task, public, 'INITIALIZE')
        return self.cached

    def snapshot(self, *, force: bool=False) -> PlanningStateView:
        if self.cached is None:
            return self.initialize()
        task = self.env.get_task_state()
        public = self.env.get_agent_state()
        transition = bool(task.target_found) != bool(self.cached.target_found)
        handoff = bool(self.refresh_on_executor_handoff and (not self._cached_executor_knows_target) and task.executor_knows_target)
        current_target = self._executor_target(task, public)
        target_shift = False
        if self.refresh_on_public_target_shift and current_target is not None and (self._last_full_refresh_executor_target is not None) and (int(task.step) - self._last_full_refresh_step >= self.public_target_update_min_steps):
            delta = sum(((float(left) - float(right)) ** 2 for (left, right) in zip(current_target, self._last_full_refresh_executor_target))) ** 0.5
            target_shift = delta >= self.public_target_update_distance
        periodic = bool(self.refresh_interval > 0 and task.step % self.refresh_interval == 0)
        refresh = force or handoff or target_shift or transition or periodic
        if refresh:
            self.cached = extract_planning_state(self.env)
            if force:
                reason = 'FORCE'
            elif handoff:
                reason = 'EXECUTOR_HANDOFF'
                self._handoff_forced_refresh_count += 1
            elif target_shift:
                reason = 'PUBLIC_TARGET_SHIFT'
                self._target_shift_forced_refresh_count += 1
            elif transition:
                reason = 'TARGET_FOUND_TRANSITION'
            else:
                reason = 'PERIODIC'
            self._record_full_refresh(task, public, reason)
            return self.cached
        agents = tuple((replace(old, position=tuple(public.positions[old.agent_id]), velocity=tuple(public.velocities[old.agent_id]), current_navigation_target=tuple(public.navigation_targets[old.agent_id])) for old in self.cached.agents))
        self.cached = replace(self.cached, step=int(task.step), target_found=bool(task.target_found), mission_complete=bool(task.mission_complete), agents=agents)
        self._cached_executor_knows_target = bool(task.executor_knows_target)
        self._last_snapshot_full_refresh = False
        self._last_refresh_reason = 'INCREMENTAL'
        return self.cached
