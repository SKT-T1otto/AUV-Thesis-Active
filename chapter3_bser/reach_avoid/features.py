"""Causal public context with obstacle DIRECTIONS; no hidden-world fields."""
from collections import deque
from dataclasses import dataclass
from itertools import product
import numpy as np
from chapter3_bser.reach_avoid.options import FRAME_DIM, SPECS
from core.env.target_motion import TargetState, predict_target_state

DIRECTIONS = np.array([x for x in product((-1., 0., 1.), repeat=3) if any(x)], float)
DIRECTIONS /= np.linalg.norm(DIRECTIONS, axis=1, keepdims=True)


@dataclass
class Packet:
    step: int
    features: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray
    flow: np.ndarray
    phases: np.ndarray
    intents: np.ndarray
    target_velocity: np.ndarray
    clearances: np.ndarray
    origin: np.ndarray
    spacing: np.ndarray
    shape: np.ndarray
    free: np.ndarray
    occupied: np.ndarray
    bounds: np.ndarray
    goals: np.ndarray | None = None
    goal_mask: np.ndarray | None = None
    actions: np.ndarray | None = None
    action_mask: np.ndarray | None = None
    checked: np.ndarray | None = None
    pose_free: np.ndarray | None = None


class Encoder:
    def __init__(self, options, max_steps):
        self.options = options
        self.max_steps = max_steps
        self.history = [deque(maxlen=options['history']) for _ in range(4)]

    def build(self, state, task, observations, semantic, geometry, guard, delivered=None):
        p = np.array([a.position for a in state.agents], float)
        v = np.array([a.velocity for a in state.agents], float)
        bounds = np.asarray(guard.contract['bounds'], float)
        known = bool(task.executor_knows_target)
        intent = np.array([semantic[i] for i in range(4)], float)
        target_velocity = np.zeros(3)
        prediction_fallback=False
        if known:
            if delivered is None or delivered.get('schema')!='public_delivered_target_state_v1':
                raise ValueError('committed public handoff payload required after target receipt')
            if delivered.get('physical_step')!=int(state.step) or not 0<=delivered.get('received_step',-1)<=int(state.step):
                raise ValueError('public handoff delivery clock mismatch')
            target=TargetState.from_payload(delivered['payload'])
            age=int(state.step)-int(target.sample_step)
            if age<0:raise ValueError('public handoff sample is in the future')
            obstacles=[dict(center=((lo+hi)/2).tolist(),size=(hi-lo).tolist())
                       for lo,hi in zip(geometry.raw.occupied_lower,geometry.raw.occupied_upper)]
            kwargs=dict(clearance=guard.contract['target_obstacle_clearance'],
                        max_reflections=guard.contract['target_max_reflections_per_step'],
                        max_prediction_steps=self.max_steps+100)
            try:
                predicted=predict_target_state(target,age,guard.contract['dt'],bounds,obstacles,**kwargs)
            except (ValueError,RuntimeError):
                prediction_fallback=True
                kwargs['clearance']=0.
                predicted=predict_target_state(target,age,guard.contract['dt'],bounds,(),**kwargs)
            target_velocity=np.asarray(predicted.velocity,float)
            intent[3]=np.asarray(predicted.position,float)
        phases = np.array([3 if state.target_found else 0] * 3 + [2 if known else 1], int)
        frames = []
        gaps = []
        radius = guard.contract['obstacle_sensor_range']
        for i, spec in enumerate(SPECS):
            rays = np.ones(26)
            points = geometry.directional_points(p[i])
            if len(points):
                delta = points - p[i]
                distances = np.linalg.norm(delta, axis=1)
                nearby = (distances <= radius) & (distances > 1e-8)
                if nearby.any():
                    sectors=np.argmax((delta[nearby]/distances[nearby,None])@DIRECTIONS.T,axis=1)
                    np.minimum.at(rays,sectors,distances[nearby]/radius)
                if np.any(distances <= 1e-8):
                    rays[:] = 0.
            cell = np.floor((p[i] - geometry.origin) / geometry.spacing).astype(int)
            cell = np.clip(cell, 0, geometry.shape - 1)
            slices = tuple(slice(max(0, int(x) - 1), min(int(n), int(x) + 2)) for x, n in zip(cell, geometry.shape))
            local = [geometry.unknown[slices].mean(), geometry.free[slices].mean(),
                     np.asarray(state.occupancy.occupied_mask).reshape(tuple(geometry.shape))[slices].mean()]
            gap = min(radius, geometry.distance(p[i]));gaps.append(gap)
            frame = np.concatenate((p[i] / bounds, v[i] / [spec['v_xy_max'], spec['v_xy_max'], spec['v_z_max']], guard.flow[i],
                np.eye(4)[i], np.eye(4)[phases[i]], (intent[i] - p[i]) / bounds, target_velocity / 3.,
                ((p - p[i]) / bounds).ravel(), (v / 3.).ravel(), rays, local,
                [state.step / self.max_steps, max(0, state.step - guard.evidence.last_scan_step) / 20.],
                [spec[k] for k in ('a_xy_max','a_z_max','v_xy_max','v_z_max','drag_xy','drag_z','buoyancy_bias')],
                guard.previous[i], [min(radius, gap) / radius, float(known), float(prediction_fallback)]))
            if frame.shape != (FRAME_DIM,) or not np.isfinite(frame).all():
                raise ValueError('invalid goal-independent public frame')
            self.history[i].append(frame.astype(np.float32))
            while len(self.history[i]) < self.options['history']:
                self.history[i].appendleft(frame.astype(np.float32))
            frames.append(np.concatenate(tuple(self.history[i])))
        return Packet(int(state.step), np.asarray(frames, np.float32), p, v, guard.flow.copy(), phases, intent,
            target_velocity, np.array(gaps), geometry.origin.copy(), geometry.spacing.copy(), geometry.shape.copy(),
            geometry.free.ravel().copy(), np.asarray(state.occupancy.occupied_mask, bool).copy(), bounds,
            pose_free=np.array([geometry.local(point)[1] for point in p], bool))
