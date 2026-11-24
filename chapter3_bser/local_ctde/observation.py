"""The simulator-to-boat boundary. Nothing here reads a fused planning map."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from core.env.public_events import PublicEvent, ScanObservation, payload_digest
from core.env.task_state import AgentStateView, MissionTaskState
from core.env.observation_contract import ROLE_ORDER
from core.mapping.path_planner import OnlineUnknownMapTaskPlanner


@dataclass(frozen=True)
class SensorFrame:
    step: int
    position: tuple
    velocity: tuple
    stopped: bool
    scans: tuple
    delivery: dict | None


def sensor_frame(agent, step, kinematics, finished, scan_events, delivery):
    """Demultiplex at the simulator boundary, before calling the on-board code."""
    scans = []
    for event in scan_events:
        if event.kind != 'scan':
            continue
        scan = event.payload
        if scan.agent_ids.count(agent) > 1 or event.physical_step > step:
            raise ValueError('invalid time or duplicate sensor row')
        if agent in scan.agent_ids:
            j = scan.agent_ids.index(agent)
            scans.append((event.physical_step, ScanObservation(
                (scan.origins[j],), scan.directions, (scan.distances[j],),
                (scan.hit_mask[j],), (agent,))))
    return SensorFrame(step, tuple(kinematics.positions[agent]), tuple(kinematics.velocities[agent]),
                       bool(finished[agent]), tuple(scans), delivery if agent == 3 else None)


class OnboardMap:
    """An empty native planner with local scans and own-state-only query endpoints.

    The legacy graph/route API takes four fixed role slots. The other positions
    are copies of the observer's pose, never supplied peer measurements.
    """

    def __init__(self, agent, planner, specs):
        if not isinstance(planner, OnlineUnknownMapTaskPlanner):
            raise ValueError('local execution requires a fresh online unknown map')
        planner.reset()
        self.agent = agent
        self.unwrapped = SimpleNamespace(map_module=planner, agent_specs=specs)
        self.role_order = ROLE_ORDER
        self.scan_count = 0
        self.last_scan_step = -1
        self.last_scan = None
        self.frame = None
        self.goal = None

    def receive(self, frame):
        if self.agent != 3 and frame.delivery is not None:
            raise ValueError('handoff payload is addressed to the executor')
        if self.frame is not None and frame.step != self.frame.step + 1:
            raise ValueError('on-board observations require consecutive physical steps')
        self.frame = frame
        if self.goal is None:
            self.goal = frame.position
        planner = self.unwrapped.map_module
        planner.set_mapping_step(frame.step)
        local_events = []
        for step, scan in frame.scans:
            if scan.agent_ids != (self.agent,) or not self.last_scan_step <= step <= frame.step:
                raise ValueError('invalid on-board scan identity or time')
            planner.integrate_obstacle_scan(scan.origins, scan.directions, scan.distances,
                                            scan.hit_mask, current_step=step)
            self.last_scan_step, self.last_scan = step, scan
            self.scan_count += 1
            local_events.append(PublicEvent('onboard', step, self.scan_count, 'own_sonar',
                                            'scan', scan, payload_digest(scan)))
        return local_events

    def get_agent_state(self):
        f = self.frame
        navigation = [f.position] * 4
        navigation[self.agent] = tuple(self.goal)
        return AgentStateView(self.role_order, (f.position,) * 4, (f.velocity,) * 4,
                              tuple(navigation), (False,) * 4)

    def get_task_state(self):
        f = self.frame
        # A local stop is the native mission's search-stop signal. The executor
        # enters tracking only on delivery. It never sees a team's found flag.
        return MissionTaskState(f.step, self.agent < 3 and f.stopped, -1,
                                f.delivery is not None, None, False, None)

    def get_scenario_identity(self):
        return {'obstacle_knowledge_mode': 'online_unknown'}


def local_features(view, bounds, horizon, launch, commitment, sensor_range):
    f, agent = view.frame, view.agent
    p, v = np.asarray(f.position), np.asarray(f.velocity)
    spec = view.unwrapped.agent_specs[agent]
    goal = np.asarray(commitment.endpoint) if commitment is not None else p
    progress = 0.0 if commitment is None else commitment.route.index / max(1, len(commitment.route.points) - 1)
    rays = np.ones(6)
    if view.last_scan is not None:
        scan = view.last_scan
        directions = np.asarray(scan.directions)
        axes = np.vstack((np.eye(3), -np.eye(3)))
        indices = (axes @ directions.T).argmax(axis=1)
        rays = np.clip(np.asarray(scan.distances[0])[indices] / sensor_range, 0., 1.)
    speed_limits = np.array([spec['v_xy_max'], spec['v_xy_max'], spec['v_z_max']])
    vector = np.r_[p / bounds, v / speed_limits,
        (goal - p) / bounds, (p - launch) / bounds, rays, np.eye(4)[agent],
        (horizon - f.step) / horizon,
        (f.step - view.last_scan_step) / horizon if view.last_scan_step >= 0 else 1.,
        float(commitment is not None), progress, float(f.stopped), float(f.delivery is not None)]
    if vector.shape != (28,) or not np.isfinite(vector).all():
        raise ValueError('invalid 28-dimensional on-board policy input')
    return vector.astype(np.float32)
