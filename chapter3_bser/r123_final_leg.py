"""Optional terminal connection for the Chapter 3 synchronized rule simulator."""
import numpy as np

from chapter3_bser.r123_repair.lifecycle import ProgressManager
from chapter3_bser.r123_task_sync.diagnostics import ObservedRuntime


def connection_verdict(commitment, position, geometry):
    """Classify the last leg of an already accepted, reachable grid route."""
    route = commitment.route
    if (commitment.mode != 'TASK' or not route.reachable or route.failure
            or not len(route.points)):
        return 'grid_required'
    if (not np.allclose(route.points[-1], commitment.endpoint, rtol=0, atol=1e-4)
            or not np.allclose(commitment.task, commitment.endpoint, rtol=0, atol=1e-4)):
        return 'different_endpoint'
    if route.index == len(route.points) - 1:
        return 'already_terminal'
    point = route.points[-1]
    # One horizontal grid spacing, fixed from the public map before evaluation.
    if np.linalg.norm(point - position) > max(geometry.spacing[:2]):
        return 'outside_final_leg'
    if not geometry.raw.segment_free(position, point):
        return 'occupied_grid'
    if not geometry.segment(position, point):
        return 'scan_veto'
    return 'direct'


class FinalLegManager(ProgressManager):
    """Keep ordinary grid progress separate from the cursor used for one action."""
    def __init__(self, options, lifecycle, r2):
        super().__init__(options, lifecycle, r2)
        self.geometry = None
        self.saved_cursor = None
        self.verdict = 'grid_required'

    def restore_grid(self):
        if self.saved_cursor is not None:
            route, index = self.saved_cursor
            route.index = index
            self.saved_cursor = None
        self.verdict = 'grid_required'

    def status(self, agent, position, velocity, phase, step, endpoint_free, target_velocity=None):
        if agent == 3 and phase == 2:
            self.restore_grid()
            task = self.lifecycle.tasks[agent]
            if endpoint_free and not task.released and task.phase == phase:
                c = self.current[agent]
                self.verdict = connection_verdict(c, position, self.geometry)
                if self.verdict == 'direct':
                    self.saved_cursor = (c.route, c.route.index)
                    c.route.index = len(c.route.points) - 1
        # Status, progress, action generation and the trace see the same waypoint.
        return super().status(agent, position, velocity, phase, step, endpoint_free, target_velocity)


class RuleRuntime(ObservedRuntime):
    def __init__(self, *args, direct_final_leg=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.direct_final_leg = direct_final_leg
        self.connection_steps = {}
        if direct_final_leg:
            self.manager = FinalLegManager(self.o, self.lifecycle, self.switches['r2'])

    def _query(self, i, point, state):
        if i == 3 and self.direct_final_leg:
            # A shortcut never certifies the skipped grid prefix as traversed.
            self.manager.restore_grid()
        return super()._query(i, point, state)

    def _options(self, i, state, packet):
        if i != 3 or not self.direct_final_leg:
            return super()._options(i, state, packet)
        manager = self.manager
        manager.restore_grid()
        manager.geometry = self._geometry
        result = super()._options(i, state, packet)
        if packet.phases[i] == 2:
            c, _, reason, _, _ = result
            if self.lifecycle.tasks[i].released or reason in ('WAIT_TASK', 'CONNECTOR'):
                manager.restore_grid()
            grid_index = manager.saved_cursor[1] if manager.saved_cursor else c.route.index
            self.connection_steps[state.step] = dict(
                step=state.step, verdict=manager.verdict, route_id=c.route.route_id,
                grid_index=grid_index, selected_index=c.route.index, map_revision=state.map_revision)
        return result

    def summary(self):
        result = super().summary()
        result['final_leg'] = dict(enabled=self.direct_final_leg, steps=[
            row for step, row in self.connection_steps.items() if step < self.step])
        return result
