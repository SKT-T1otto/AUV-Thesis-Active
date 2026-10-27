"""Role/phase-conditioned goal sets and factual first-event semantics."""
from dataclasses import dataclass
import numpy as np
from chapter3_bser.safe_rl.dynamics import prior, limits
from chapter3_bser.reach_avoid.options import SPECS, CONTRACT

SEARCH, STANDBY, CAPTURE, VIA = range(4)


@dataclass(frozen=True)
class Goal:
    center: tuple
    velocity: tuple = (0., 0., 0.)
    kind: int = SEARCH
    reference: tuple | None = None

    @property
    def ref(self):
        return self.center if self.reference is None else self.reference

    def array(self):
        return np.concatenate((self.center, self.velocity, np.eye(4)[self.kind], self.ref)).astype(np.float32)

    @classmethod
    def from_array(cls, value):
        x = np.asarray(value, float)
        if x.shape != (13,) or not np.isfinite(x).all() or not np.array_equal(x[6:10], np.eye(4)[np.argmax(x[6:10])]):
            raise ValueError('invalid goal representation')
        return cls(tuple(x[:3]), tuple(x[3:6]), int(np.argmax(x[6:10])), tuple(x[10:13]))


def cell_status(packet, point):
    p = np.asarray(point, float)
    cell = np.floor((p - packet.origin) / packet.spacing).astype(int)
    inside = bool(np.all(cell >= 0) and np.all(cell < packet.shape))
    if not inside:
        return False, False, False
    index = np.ravel_multi_index(tuple(cell), tuple(packet.shape))
    return bool(packet.free[index]), bool(packet.occupied[index]), True


def base_action(packet, agent, goal):
    """Shared rule baseline: original PD plus PUBLIC estimated disturbance/goal velocity.

    Both rule and learned comparisons use this exact base. Compensation is not
    claimed as the learned contribution, and parent D2 is never changed.
    """
    spec = SPECS[agent]
    u = prior(packet.positions[agent], packet.velocities[agent], goal.ref, spec, CONTRACT)
    u += np.array([CONTRACT['prior_kv_xy']] * 2 + [CONTRACT['prior_kv_z']]) * np.asarray(goal.velocity)
    u[:2] -= packet.flow[agent] / (1 - spec['drag_xy'] * CONTRACT['dt'])
    u[2] -= spec['buoyancy_bias'] / (1 - spec['drag_z'] * CONTRACT['dt'])
    return np.clip(u / limits(spec), -1., 1.).astype(np.float32)


def encode_goal(packet, agent, goal):
    free, occupied, inside = cell_status(packet, goal.center)
    scale = np.array([SPECS[agent]['v_xy_max']] * 2 + [SPECS[agent]['v_z_max']])
    return np.concatenate(((np.asarray(goal.center) - packet.positions[agent]) / packet.bounds,
        np.asarray(goal.velocity) / scale, np.eye(4)[goal.kind], [free, occupied, inside],
        (np.asarray(goal.ref) - packet.positions[agent]) / packet.bounds,
        base_action(packet, agent, goal))).astype(np.float32)


def goal_reached(packet, agent, goal, options):
    """Local auxiliary event, never a replacement for environment capture/success."""
    free_pose, _, inside_pose = cell_status(packet, packet.positions[agent])
    if not inside_pose or not free_pose or packet.clearances[agent] <= options['clearance'] + options['hit_padding']:
        return False
    radius = options['capture_radius'] if goal.kind == CAPTURE else options['goal_radius']
    if np.linalg.norm(packet.positions[agent] - np.asarray(goal.center)) > radius:
        return False
    if goal.kind != CAPTURE and not cell_status(packet, goal.center)[0]:
        return False
    if goal.kind == STANDBY:
        return bool(np.linalg.norm(packet.velocities[agent]) <= options['hold_speed'])
    if goal.kind == CAPTURE:
        return bool(np.linalg.norm(packet.velocities[agent] - goal.velocity) <= options['hold_speed'])
    return True


def next_goal(goal, packet, agent):
    if goal.kind != CAPTURE:
        return goal
    # The goal tracks only the handed-off public estimate reconstructed from the
    # committed delivery, never a hidden target state or a true future path.
    center = packet.intents[agent]
    offset = np.asarray(goal.ref) - goal.center
    return Goal(tuple(center), tuple(packet.target_velocity), CAPTURE, tuple(center + offset))
