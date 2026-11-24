"""One on-board instance of the R123 no-recovery navigation rules."""
import numpy as np

from core.mapping.planning_state import extract_planning_state
from chapter3_bser.integration.control_context import AgentAssignmentContextV1
from chapter3_bser.path_option.options import validate as path_options
from core.mapping.travel_cost_service import TravelCostService
from chapter3_bser.r123.options import CONTROL
from chapter3_bser.r123_final_leg import FinalLegManager, RuleRuntime
from chapter3_bser.r123_repair.lifecycle import Lifecycle
from chapter3_bser.r123_tracking.control import conditional_banks
from chapter3_bser.r123_tracking.progress import tracking_velocity
from chapter3_bser.reach_avoid.features import Encoder
from chapter3_bser.reach_avoid.scan_geometry import RaySubcells, ScanRefinedGeometry
from chapter3_bser.safe_rl.control import SafetyController
from chapter3_bser.safe_rl.navigation import EndpointView
from chapter3_bser.safe_rl.options import validate as guard_options
from chapter3_bser.safety.planning_refresh import PlanningViews, start_status
from .observation import local_features


class LocalNavigator(RuleRuntime):
    """Reuse the original single-agent _options/_query call chain.

    Construct only its on-board dependencies. The parent episode constructor
    would create a team allocator and perform a joint allocation on reset.
    """

    def __init__(self, view, contract, options, horizon, clearance, standby=None):
        self.env, self.agent = view, view.agent
        if (self.agent == 3) != (standby is not None):
            raise ValueError('only the executor receives a preflight standby coordinate')
        self.horizon = horizon
        self.bounds = np.array(contract['bounds'])
        self.point = None if standby is None else np.array(standby, dtype=float)
        self.launch = None
        self.o = path_options(options)
        self.variant = 'no_recovery'
        self.direct_final_leg = True
        self.tracking_switches = dict(path=True, feedforward=True)
        self.switches = dict(r1=True, r2=True, r3=True)
        self.lifecycle = Lifecycle(CONTROL)
        self.manager = FinalLegManager(self.o, self.lifecycle, True)
        self.safe_options = dict(path_safety=True)
        self.planning_views = PlanningViews(self, clearance)
        self.guard = SafetyController(view.unwrapped.agent_specs, contract, guard_options())
        self.encoder = Encoder(dict(history=1), horizon)
        self.subcells = RaySubcells(contract['bounds'])
        self._release_requests = set()
        self._request_refresh = False
        self._last_request = -CONTROL['request_interval']
        self._task_tokens, self._connectors, self._last_root_query = {}, {}, {}
        self._candidates, self._alternatives, self._last_candidates = {}, {}, {}
        self.sync_route_queries = 0
        self.route_events, self.connection_steps = [], {}
        self.proposal_queries = 0

    def assignment(self, i):
        if i != self.agent:
            raise ValueError('another boat has no assignment in this navigator')
        point = self.point
        return dict(assigned=point is not None, token=None if point is None else tuple(point),
                    point=point, kind='search' if i < 3 else 'executor')

    def _original(self, i, step):
        return np.array(self.env.frame.position if self.point is None else self.point)

    def sense(self, frame):
        events = self.env.receive(frame)
        if self.launch is None:
            self.launch = np.array(frame.position)
        c = self.manager.current.get(self.agent)
        task = self.lifecycle.tasks.get(self.agent)
        if task is None or task.released:
            c = None
        self.features = local_features(self.env, self.bounds, self.horizon, self.launch, c,
                                      self.guard.contract['obstacle_sensor_range'])
        self.state = extract_planning_state(self.env)
        self.guard.observe(self.state, events)
        self.subcells.consume(events, frame.step)
        self._geometry = ScanRefinedGeometry(self.state, self.guard.evidence, self.guard.options, self.subcells)
        return self.features

    def _can_plan_to(self, point):
        self.proposal_queries += 1
        state = extract_planning_state(EndpointView(self.env, self.agent, point))
        state = self.planning_views.search(state)
        agent = state.agents[self.agent]
        return TravelCostService(state).query(agent.position, point, agent).reachable

    def _project(self, raw):
        # Keep continuous proposals when reachable. For a blocked proposal try
        # the closest valid reachable-component cell once. No team reranking.
        i, state = self.agent, self.planning_views.search(self.state)
        if start_status(state, i) != 'ready':
            return None
        if self._can_plan_to(raw):
            return raw.copy()
        graph = state.planning_graph
        start = np.asarray(self.env.frame.position)
        endpoint = next(e for e in graph.endpoint_connectors
                        if e.role == 'searcher' and np.allclose(e.point, start, rtol=0, atol=1e-12))
        labels = graph.component_labels
        components = [labels[edge.cell_index] for edge in endpoint.connectors]
        valid = np.flatnonzero(graph.valid_mask & np.isin(labels, components))
        if not len(valid):
            return None
        centers = graph.cell_centers[valid]
        projected = centers[np.argmin(np.linalg.norm(centers - raw, axis=1))]
        return projected.copy() if self._can_plan_to(projected) else None

    def act(self, policy, generator=None):
        i, f = self.agent, self.env.frame
        decision = None
        if (i < 3 and not f.stopped and (i not in self.lifecycle.tasks or i in self._release_requests)
                and f.step - self._last_request >= CONTROL['request_interval']):
            self._last_request = f.step
            latent, unit = policy.propose(self.features, generator)
            raw = (unit + 1.) * .5 * self.bounds
            self.point = self._project(raw)
            decision = dict(agent=i, step=f.step, features=self.features.tolist(),
                            latent=latent.tolist(), raw_goal=raw.tolist(),
                            proposed=None if self.point is None else self.point.tolist())
        self.env.goal = self._original(i, f.step)
        semantic = {k: f.position for k in range(4)}
        semantic[i] = self.env.goal
        packet = self.encoder.build(self.state, self.env.get_task_state(), None, semantic,
                                    self._geometry, self.guard, f.delivery)
        c, _, reason, _, _ = self._options(i, self.state, packet)
        route = c.route
        track = route.tracking(f.position)
        phase = int(packet.phases[i])
        velocity = tracking_velocity(packet, i, route, c.endpoint, True)
        bank, masks, forces, _, diagnostics = conditional_banks(
            packet, i, [track], self._geometry, self.guard, target_velocities=[velocity])
        mask, forced = masks[0], int(forces[0])
        if phase == 3:
            command = np.zeros(3, dtype=np.float32)
        else:
            if reason == 'WAIT_TASK':
                forced = 9 if diagnostics[0]['verified_exit'] else 8
            elif reason in ('HOLD', 'SETTLING') and forced < 0:
                forced = int(np.flatnonzero(mask)[0])
            choice = forced
            if choice < 0:
                choice = 0 if mask[0] else 8 if mask[8] else int(np.flatnonzero(mask)[0])
            command = bank[0, choice].copy()
        self.guard.previous_v = packet.velocities.copy()
        task = self.lifecycle.tasks[i]
        if decision is not None:
            decision['accepted'] = bool(self.point is not None and not task.released
                                        and np.allclose(self.point, task.point, rtol=0, atol=1e-6))
            decision['root'] = task.point.tolist()
            decision['route_id'] = route.route_id
        item = AgentAssignmentContextV1(
            agent_id=i, role=self.env.role_order[i], assignment_kind=('search', 'standby', 'execution', 'hold')[phase],
            assignment_id=str(task.identity), final_waypoint=tuple(c.endpoint),
            planned_path=tuple(map(tuple, route.points)), tracking_waypoint=tuple(track),
            hold_position=tuple(c.endpoint), hold_state=phase == 3 or reason in ('HOLD', 'SETTLING'),
            reachable=bool(route.reachable), execution_request=phase == 2)
        return command, item, decision

    def own_applied_control(self, acceleration):
        self.guard.previous[self.agent] = np.asarray(acceleration)
