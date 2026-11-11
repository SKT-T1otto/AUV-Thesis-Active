"""Public D2 route queries and persistent paths. No planner refresh or truth access."""
from dataclasses import dataclass, field
import hashlib
import numpy as np
from core.mapping.planning_state import extract_planning_state
from core.mapping.travel_cost_service import TravelCostService
from chapter3_bser.safe_rl.navigation import EndpointView

PATH_DIM = 13


def public_route(env, agent_id, goal, *, query_start=None):
    class QueryView(EndpointView):
        def get_agent_state(self):
            state = super().get_agent_state()
            if query_start is None:
                return state
            from dataclasses import replace
            positions = list(state.positions); positions[self.agent_id] = tuple(query_start)
            return replace(state, positions=tuple(positions))
    state = extract_planning_state(QueryView(env, agent_id, goal))
    agent = next(a for a in state.agents if a.agent_id == agent_id)
    return TravelCostService(state).query(agent.position, goal, agent)


@dataclass
class Route:
    points: np.ndarray
    revision: int
    query_step: int
    reachable: bool
    failure: str = ''
    index: int = 0
    distance: float = 0.
    estimated_seconds: float = 0.
    route_id: str = field(init=False)

    def __post_init__(self):
        self.points = np.asarray(self.points, np.float32).reshape(-1, 3).copy()
        self.route_id = hashlib.sha256(self.points.tobytes()).hexdigest()[:16]

    def advance(self, position, radius):
        # Only reach tests advance progress. A new map counter alone never resets it.
        while self.index < len(self.points)-1 and np.linalg.norm(self.points[self.index]-position) <= radius:
            self.index += 1
        return self.tracking(position)

    def tracking(self, position):
        return self.points[min(self.index, len(self.points)-1)].copy() if len(self.points) else np.asarray(position).copy()

    def tokens(self, state, position, bounds):
        # ALL remaining vertices are retained; padding is done only in the batch.
        points = self.points[self.index:]
        if not len(points):
            points = np.asarray(position).reshape(1, 3)
        delta = np.diff(points, axis=0, append=points[-1:])
        tangent = delta / np.maximum(np.linalg.norm(delta, axis=1, keepdims=True), 1e-6)
        turn = np.linalg.norm(np.diff(tangent, axis=0, prepend=tangent[:1]), axis=1)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        cells = np.floor((points-np.asarray(state.grid.origin))/state.grid.spacing).astype(int)
        cells = np.clip(cells, 0, np.asarray(state.grid.shape)-1)
        ids = np.ravel_multi_index(cells.T, state.grid.shape)
        occ = state.occupancy
        # Per-cell scan timestamps are unavailable: expose snapshot age, not invented ages.
        return np.column_stack(((points-position)/bounds, tangent,
            arc/max(float(np.linalg.norm(bounds)), 1.), turn,
            np.asarray(occ.occupancy_probability)[ids], np.asarray(occ.unknown_mask)[ids],
            np.full(len(points), max(0, state.step-self.query_step)/100.),
            np.linspace(0., 1., len(points)), np.full(len(points), float(self.reachable)))).astype(np.float32)

    def cross_track(self, position):
        points = self.points[max(0, self.index-1):]
        if len(points) < 2:
            return float(np.linalg.norm(self.tracking(position)-position))
        p = np.asarray(position); a = points[:-1]; d = np.diff(points, axis=0)
        t = np.clip(np.sum((p-a)*d, axis=1)/np.maximum(np.sum(d*d, axis=1), 1e-12), 0., 1.)
        return float(np.linalg.norm(p-(a+t[:, None]*d), axis=1).min())


@dataclass
class Commitment:
    original: np.ndarray
    task: np.ndarray
    endpoint: np.ndarray
    received: int
    phase: int
    goal_id: int
    route: Route
    mode: str = 'TASK'
    best_distance: float = float('inf')
    progress_step: int = 0
    ended_reason: str = 'INITIAL'


class GoalManager:
    def __init__(self, options):
        self.options = options
        self.current = {}
        self.serial = 0

    def commit(self, agent, original, task, endpoint, step, phase, route, *, reason):
        self.serial += 1
        c = Commitment(np.array(original).copy(), np.array(task).copy(), np.array(endpoint).copy(),
            step, int(phase), self.serial, route,
            'TASK' if np.linalg.norm(np.asarray(endpoint)-task) < 1e-4 else 'RECOVER',
            progress_step=step, ended_reason=reason)
        self.current[agent] = c
        return c

    def status(self, agent, position, velocity, phase, step, endpoint_free, target_velocity=None):
        c = self.current.get(agent)
        if c is None:
            return 'INITIAL'
        if phase != c.phase:
            return 'PHASE_CHANGED'
        if not endpoint_free:
            return 'KNOWN_INVALID'
        distance = float(np.linalg.norm(position-c.endpoint))
        if distance < c.best_distance-self.options['progress_distance']:
            c.best_distance, c.progress_step = distance, step
        if distance <= self.options['arrival_radius']:
            if phase in (1, 2):
                relative = velocity if target_velocity is None else velocity-target_velocity
                return 'HOLD' if np.linalg.norm(relative) <= .2 else 'SETTLING'
            return 'REGION_ARRIVED'
        if not c.route.reachable:
            return 'NO_ROUTE'
        return 'STALLED' if step-c.progress_step >= self.options['stagnation_steps'] else 'CONTINUE'

    def rule_terminate(self, reason, age, *, periodic=False):
        if reason in ('INITIAL', 'PHASE_CHANGED', 'REGION_ARRIVED', 'KNOWN_INVALID', 'NO_ROUTE'):
            return True
        if reason in ('HOLD', 'SETTLING'):
            return False
        return age >= self.options['periodic_steps'] if periodic else reason == 'STALLED' and age >= self.options['minimum_commitment']

    def termination_flags(self, reason, age, phase, has_choices):
        if not has_choices or phase == 3:
            return False, False
        hard = reason in ('INITIAL', 'PHASE_CHANGED', 'REGION_ARRIVED', 'KNOWN_INVALID', 'NO_ROUTE')
        allowed = age >= self.options['minimum_commitment'] and reason in ('CONTINUE', 'STALLED')
        return hard, allowed
