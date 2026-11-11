"""Persistent OLD tasks, public D2 routes, common safety bank and factual replay."""
import copy
from dataclasses import replace
import numpy as np
import torch
from core.mapping.planning_state import extract_planning_state
from chapter3_bser.learning.runtime import RepairedMissionRuntime
from chapter3_bser.safe_rl.runtime import SafetyRuntime
from chapter3_bser.safe_rl.control import SafetyController
from chapter3_bser.safe_rl.options import validate as safety_options
from chapter3_bser.safe_rl.dynamics import brake, limits, integrate
from chapter3_bser.reach_avoid.features import Encoder as PublicEncoder
from chapter3_bser.reach_avoid.options import SPECS, CONTRACT
from chapter3_bser.reach_avoid.scan_geometry import RaySubcells, ScanRefinedGeometry, refined_buffer_exit
from chapter3_bser.reach_avoid.goals import Goal, base_action
from chapter3_bser.path_option.path_context import GoalManager, Route, public_route
from chapter3_bser.path_option.options import validate, SCHEMA


def route_from_query(result, revision, step):
    return Route(result.path_points, revision, step, bool(result.reachable), result.failure_reason or '',
        distance=float(result.path_length) if result.reachable else 0.,
        estimated_seconds=float(result.physical_travel_time) if result.reachable else 0.)


def observed_occupied(state, point):
    cell = np.floor((np.asarray(point)-state.grid.origin)/state.grid.spacing).astype(int)
    if np.any(cell < 0) or np.any(cell >= state.grid.shape):
        return True
    return bool(np.asarray(state.occupancy.occupied_mask)[np.ravel_multi_index(cell, state.grid.shape)])


def make_bank(packet, i, tracks, geometry, guard):
    """Same public action/safety construction in every new variant, including rule."""
    p, v, spec = packet.positions[i], packet.velocities[i], SPECS[i]
    candidates = [base_action(packet, i, Goal(tuple(t), tuple(packet.target_velocity) if packet.phases[i] == 2 else (0., 0., 0.))) for t in tracks]
    b = brake(v, spec, CONTRACT['dt'], packet.flow[i])/limits(spec)
    recovery = refined_buffer_exit(p, v, spec, geometry, guard.contract, guard.options, packet.flow[i])
    candidates.extend([b, b if recovery is None else recovery[0]/limits(spec)])
    for axis in range(3):
        for sign in (-1, 1):
            candidates.append(np.clip(b+np.eye(3)[axis]*.5*sign, -1, 1))
    cap = guard._caps(p, v, spec, geometry)
    violation = max(0., np.linalg.norm(v[:2])-cap[0])+max(0., abs(v[2])-cap[1])
    valid = []
    for k, a in enumerate(candidates):
        if k == 9:
            ok = recovery is not None
        else:
            nv = integrate(p, v, a*limits(spec), spec, CONTRACT['dt'], packet.flow[i])[1]
            nv_bad = max(0., np.linalg.norm(nv[:2])-cap[0])+max(0., abs(nv[2])-cap[1])
            ok = (nv_bad <= 1e-6 or nv_bad < violation-1e-8) and guard.check(p, v, a*limits(spec), spec, geometry, packet.flow[i])[0]
        valid.append(bool(ok))
    emergency = not any(valid)
    if emergency:
        valid[8] = True
    return np.asarray(candidates, np.float32), np.array(valid), 9 if recovery is not None else 8 if emergency else -1, emergency


class PathOptionRuntime(SafetyRuntime):
    def __init__(self, config, scenario, *, seed, episode_id=0, options=None, learner=None,
                 intervention='none', faults=None):
        super().__init__(config, scenario, seed=seed, episode_id=episode_id, collection_only=True)
        self.o = validate(options)
        self.learner = learner
        self.intervention = intervention
        self.manager = GoalManager(self.o)
        self.guard = SafetyController(self.env.get_public_vehicle_specs(), self.nominal_contract, safety_options())
        self.encoder = PublicEncoder(dict(history=self.o['history']), config['max_steps'])
        self.subcells = RaySubcells(self.nominal_contract['bounds'])
        self.faults = faults or []
        self.trace = []
        self.transitions = []
        self._cached = None
        self._candidates = {}
        self._alternatives = {}
        self._last_candidates = {}
        self._candidate_goal_ids = {}
        self._requested_goal_ids = set()
        self._request_refresh = False

    def _original(self, i, step):
        original = np.array(next(x.final_waypoint for x in self.guidance.agent_assignments if x.agent_id == i), float)
        for fault in self.faults:
            if fault['agent'] == i and fault['start'] <= step < fault['end']:
                original = np.array(fault['point'], float)
        return original

    def _query(self, i, point, state):
        route = route_from_query(public_route(self.env, i, point), state.map_revision, state.step)
        if not route.reachable and route.failure in ('invalid_start', 'no_start_connector'):
            # A mixed coarse cell is not a physical collision. Attach a PUBLIC
            # ray-verified escape connector without changing the original map.
            p = np.asarray(state.agents[i].position)
            centers = np.asarray(state.grid.cell_centers)
            free = np.flatnonzero(state.occupancy.free_mask)
            for k in free[np.argsort(np.linalg.norm(centers[free]-p, axis=1))][:4]:
                start = centers[k]
                if np.linalg.norm(start-p) > 3. or not self._geometry.segment(p, start):
                    continue
                q = public_route(self.env, i, point, query_start=start)
                if q.reachable:
                    route = route_from_query(q, state.map_revision, state.step)
                    route.points = np.vstack((p, route.points)).astype(np.float32)
                    route.__post_init__()
                    route.distance += float(np.linalg.norm(start-p))
                    route.estimated_seconds += float(np.linalg.norm(start-p))/SPECS[i]['v_xy_max']
                    route.failure = 'public_ray_start_connector'
                    break
        if route.reachable:
            velocity = np.asarray(state.agents[i].velocity)
            a = max(.1, SPECS[i]['a_xy_max']*(1-SPECS[i]['drag_xy']*.2)-.18)
            route.estimated_seconds += float(np.linalg.norm(velocity[:2]))/a
        return route

    def _options(self, i, state, packet):
        p = packet.positions[i]; phase = int(packet.phases[i]); now = state.step
        original = self._original(i, now)
        c = self.manager.current.get(i)
        if c is None or phase != c.phase:
            task = packet.intents[i] if phase == 2 else original
            route = self._query(i, task, state)
            c = self.manager.commit(i, original, task, task, now, phase, route, reason='INITIAL' if c is None else 'PHASE_CHANGED')
        # The handed-off moving target is a moving REGION, not a fresh task each step.
        if phase == 2:
            delta = packet.intents[i]-c.task
            if np.linalg.norm(delta) >= .75:
                c.task = packet.intents[i].copy()
                c.endpoint = c.endpoint+delta
                c.route = self._query(i, c.endpoint, state)
        c.route.advance(p, self.o['arrival_radius'])
        tracking = c.route.tracking(p)
        blocked = observed_occupied(state, tracking)
        if now-c.route.query_step >= self.o['route_query_interval'] and (blocked or not c.route.reachable):
            replacement = self._query(i, c.endpoint, state)
            c.route = replacement
            c.route.advance(p, self.o['arrival_radius'])
        reason = self.manager.status(i, p, packet.velocities[i], phase, now, not observed_occupied(state, c.endpoint),
            packet.target_velocity if phase == 2 else None)
        if reason == 'REGION_ARRIVED' and c.goal_id not in self._requested_goal_ids:
            self._request_refresh = True
            self._requested_goal_ids.add(c.goal_id)
        # Reuse alternatives between bounded query rounds; rebase starts when queried.
        if i not in self._alternatives or now-self._last_candidates.get(i, -1000) >= self.o['route_query_interval'] or (reason == 'REGION_ARRIVED' and self._candidate_goal_ids.get(i) != c.goal_id):
            centers = np.asarray(state.grid.cell_centers)
            unknown = np.asarray(state.occupancy.unknown_mask)
            free = np.asarray(state.occupancy.free_mask)
            distance = np.linalg.norm(centers-p, axis=1)
            # Detours may be farther from the task. Include spatially diverse public cells.
            task_distance = np.linalg.norm(centers-c.task, axis=1)
            score = task_distance+.15*distance-.25*unknown
            pool = []
            proposals = [original, c.task]
            proposals.extend(c.route.points[c.route.index::max(1, (len(c.route.points)-c.route.index)//3)][:3])
            near = np.flatnonzero((free | unknown) & (distance > 1.5) & (distance < 10.))
            ordered = near[np.argsort(score[near])]
            # Complement task-distance ranking with directional detours.
            for axis in range(3):
                for sign in (-1, 1):
                    if len(near):
                        proposals.append(centers[near[np.argmax(sign*(centers[near, axis]-p[axis]))]])
            proposals.extend(centers[ordered[:12]])
            # Query at most 14 endpoints per agent per planning round.
            for point in proposals[:14]:
                point = np.asarray(point, float)
                if np.linalg.norm(point-p) <= self.o['arrival_radius'] or observed_occupied(state, point):
                    continue
                if np.linalg.norm(point-c.endpoint) < 1e-4 or any(np.linalg.norm(point-x[0]) < .5 for x in pool):
                    continue
                route = self._query(i, point, state)
                if route.reachable and self.o['min_option_seconds'] <= route.estimated_seconds <= self.o['max_option_seconds']:
                    pool.append((point.copy(), route))
                if len(pool) == 7:
                    break
            self._alternatives[i] = pool
            self._last_candidates[i] = now
            self._candidate_goal_ids[i] = c.goal_id
        options = [(c.endpoint.copy(), c.route), *self._alternatives[i]][:8]
        for _, route in options:
            route.advance(p, self.o['arrival_radius'])
        mask = [bool(r.reachable and not observed_occupied(state, point)) for point, r in options]
        if phase == 0:
            mask = [ok and np.linalg.norm(point-p) > self.o['arrival_radius'] for ok, (point, _) in zip(mask, options)]
        while len(options) < 8:
            options.append((c.endpoint.copy(), c.route)); mask.append(False)
        # Current commitment may be invalid; it remains addressable for beta=1 and diagnostics.
        self._has_choices = getattr(self, '_has_choices', {})
        self._has_choices[i] = any(mask)
        if not any(mask):
            mask[0] = True
        self._candidates[i] = options
        return c, original, reason, options, np.array(mask)

    def observe(self):
        if self._cached is not None and self._cached[0] == self.step:
            return self._cached[1]
        state = extract_planning_state(self.env)
        events = self.env.read_public_events(self.guard.evidence.cursor)
        self.guard.observe(state, events); self.subcells.consume(events, state.step)
        geometry = ScanRefinedGeometry(state, self.guard.evidence, self.guard.options, self.subcells)
        self._geometry = geometry
        original = {i: self._original(i, state.step) for i in range(4)}
        packet = self.encoder.build(state, self.env.get_task_state(), self.observations, original, geometry,
                                    self.guard, self.env.get_public_delivered_target())
        rows = []
        for i in range(4):
            c, raw, reason, candidates, goal_mask = self._options(i, state, packet)
            p = packet.positions[i]; bounds = packet.bounds
            tracks = [r.tracking(p) for _, r in candidates]
            actions, action_mask, forced, emergency = make_bank(packet, i, tracks, geometry, self.guard)
            phase = int(packet.phases[i]); age = state.step-c.received
            if phase == 3:
                actions = actions.copy(); actions[8] = 0.; action_mask = np.zeros(16, bool); action_mask[8] = True; forced = 8
            elif reason in ('HOLD', 'SETTLING') and forced < 0:
                forced = 0 if action_mask[0] else int(np.flatnonzero(action_mask)[0])
            features = []
            for endpoint, route in candidates:
                features.append(np.r_[(endpoint-p)/bounds, (c.task-p)/bounds, (route.tracking(p)-p)/bounds,
                    np.eye(4)[phase], min(age, 400)/400., route.index/max(1, len(route.points)), np.eye(4)[i]])
            # Rule recovery uses PLANNED distance and time, never only Euclidean progress.
            scores = [np.linalg.norm(point-c.task)+.1*r.estimated_seconds for point, r in candidates]
            valid = np.flatnonzero(goal_mask)
            if reason == 'STALLED' and any(valid != 0):
                valid = valid[valid != 0]
            rule = int(valid[np.argmin(np.array(scores)[valid])])
            rule_end = self.manager.rule_terminate(reason, age)
            period_end = self.manager.rule_terminate(reason, age, periodic=True)
            hard, allowed = self.manager.termination_flags(reason, age, phase, self._has_choices[i])
            if self.intervention == 'rule_action' and forced < 0:
                target = actions[rule]
                forced = int(np.flatnonzero(action_mask)[np.argmin(np.linalg.norm(actions[action_mask]-target, axis=1))])
            rows.append(dict(history=packet.features[i].reshape(self.o['history'], 90).copy(),
                paths=[r.tokens(state, p, bounds) for _, r in candidates], goals=np.asarray(features, np.float32),
                actions=actions, action_mask=action_mask, goal_mask=goal_mask, forced=forced,
                event_relative=np.asarray([(t-p)/bounds for t in tracks], np.float32),
                event_targets=np.array(tracks), position=p.copy(), velocity=packet.velocities[i].copy(), bounds=bounds.copy(),
                phase=phase, reason=reason, age=age, rule_goal=rule,
                rule_terminate=rule_end, periodic_terminate=period_end, forced_terminate=hard,
                beta_allowed=allowed, intervention=self.intervention, emergency=emergency,
                original=raw.copy(), task=c.task.copy(), goal_id=c.goal_id, route_id=c.route.route_id,
                route_index=c.route.index, route_revision=c.route.revision, map_revision=state.map_revision,
                cross_track=c.route.cross_track(p), estimated_seconds=c.route.estimated_seconds,
                endpoint=c.endpoint.copy(), free_pose=bool(packet.pose_free[i]), clearance=float(packet.clearances[i])))
            rows[-1]['target_velocity'] = packet.target_velocity.copy()
            rows[-1]['committed_original'] = c.original.copy()
        self._cached = (self.step, rows)
        return rows

    def _actions_for_step(self, actions):
        RepairedMissionRuntime._actions_for_step(self, actions)
        rows = self.observe(); commands = []; self._pending = []
        assignments = {x.agent_id: x for x in self.guidance.agent_assignments}
        output = []
        for i, row in enumerate(rows):
            if self.learner is None:
                terminate = row['rule_terminate']
                j = row['rule_goal'] if terminate else 0
                target = row['actions'][j]
                k = int(np.flatnonzero(row['action_mask'])[np.argmin(np.linalg.norm(row['actions'][row['action_mask']]-target, axis=1))])
                if row['forced'] >= 0:
                    k = row['forced']
                decision = dict(goal=j, action=k, terminate=terminate, beta=float(terminate), reach=None, risk=None, task=None)
            else:
                decision = self.learner.decide(row, explore=self._explore)
            j, k = decision['goal'], decision['action']
            c = self.manager.current[i]
            if decision['terminate']:
                endpoint, route = self._candidates[i][j]
                task = row['task']
                if row['reason'] == 'REGION_ARRIVED':
                    task = row['original'] if np.linalg.norm(row['original']-row['position']) > self.o['arrival_radius'] else endpoint
                # Arrival requests a new task even if upstream D2 has not changed its proposal.
                c = self.manager.commit(i, row['original'], task, endpoint, self.step, row['phase'],
                    copy.deepcopy(route), reason=row['reason'] if row['forced_terminate'] else 'LEARNED_OR_RULE_TERMINATION')
            command = row['actions'][k].copy(); commands.append(command)
            route = c.route
            target = route.tracking(row['position'])
            item = assignments[i]
            output.append(replace(item, final_waypoint=tuple(c.endpoint), planned_path=tuple(map(tuple, route.points)),
                tracking_waypoint=tuple(target), hold_position=tuple(c.endpoint),
                hold_state=row['reason'] in ('HOLD', 'SETTLING') or row['phase'] == 3,
                reachable=route.reachable))
            self._pending.append((row, decision))
            decision['executed_commitment'] = dict(task=c.task.tolist(), endpoint=c.endpoint.tolist(),
                goal_id=c.goal_id, route_id=c.route.route_id, mode=c.mode,
                route_index=c.route.index, route_revision=c.route.revision,
                cross_track=c.route.cross_track(row['position']))
        executor = output[3]
        context = replace(self.guidance, agent_assignments=tuple(output),
            executor_assignment=replace(self.guidance.executor_assignment, target_region=executor.final_waypoint,
                planned_path=executor.planned_path, tracking_waypoint=executor.tracking_waypoint,
                hold_position=executor.hold_position, hold_state=executor.hold_state, reachable=executor.reachable))
        # Keep self.guidance as the upstream D2 proposal, distinct from installed execution context.
        self.env.install_guidance(context)
        self.guard.previous_v = np.array([r['velocity'] for r in rows])
        return torch.tensor(np.array(commands), dtype=torch.float32)

    def advance(self, model, *, explore=False):
        self._explore = explore
        if self._request_refresh and not self.terminal:
            # Explicit new task-completion event; not a diagnostic or per-step refresh.
            self.state = self.provider.snapshot(force=True)
            from chapter3_bser.learning.base_runtime import _public_context
            context = _public_context(self.env, self.state)
            result = self.controller.step(self.state, context)
            self.env.observe_controller_result(result, controller=self.controller, state_provider=self.provider)
            self.guidance = self.bridge.compile_guidance(result.allocation, self.state, context, decision_reason=result.decision_reason)
            self._request_refresh = False
        row = RepairedMissionRuntime.advance(self, model, explore=False)
        actual = self.env.unwrapped._agent_acc.detach().cpu().numpy().copy()
        self.guard.previous = actual.copy()
        actual = actual/np.array([limits(s) for s in SPECS])
        nxt = self.observe()
        collision = bool(self.env.unwrapped.last_collision_records)
        for i, (s, decision) in enumerate(self._pending):
            g = decision['goal']; target = s['event_targets'][g]
            reached = bool(s['phase'] != 3 and not collision and nxt[i]['free_pose'] and nxt[i]['clearance'] > .52 and
                np.linalg.norm(nxt[i]['position']-target) <= self.o['arrival_radius'])
            if s['phase'] in (1, 2):
                reached &= bool(np.linalg.norm(nxt[i]['velocity']-(s['target_velocity'] if s['phase'] == 2 else 0.)) <= .2)
            actual_matches = np.linalg.norm(actual[i]-s['actions'][decision['action']]) < 1e-5
            t = dict(state=s, next=nxt[i], goal_index=g, actual_action=actual[i].astype(np.float32),
                reward=float(row['team_reward']), done=bool(self.terminal), collision=collision,
                reached=reached, event_target=target.copy(), event_deadline=bool(self.terminal or s['phase'] == 3 or s['phase'] != nxt[i]['phase']),
                policy_executed=bool(s['forced'] < 0 and actual_matches and s['phase'] != 3), decision=decision['terminate'])
            # Frozen searchers still have a TEAM-return continuation value.
            # Learn that critic from factual zero-action transitions; never train their actor.
            self.transitions.append(t)
            executed = decision['executed_commitment']
            endpoint_arrived = bool(not collision and nxt[i]['free_pose'] and nxt[i]['clearance'] > .52 and
                np.linalg.norm(nxt[i]['position']-np.array(executed['endpoint'])) <= self.o['arrival_radius'])
            if s['phase'] in (1, 2):
                endpoint_arrived &= bool(np.linalg.norm(nxt[i]['velocity']-(s['target_velocity'] if s['phase'] == 2 else 0.)) <= .2)
            self.trace.append(dict(step=int(row['t']), agent=i, phase=s['phase'], original=s['original'].tolist(),
                committed_original=s['committed_original'].tolist(),
                task=executed['task'], endpoint=executed['endpoint'], goal_id=executed['goal_id'], route_id=executed['route_id'],
                route_index=executed['route_index'], route_revision=executed['route_revision'], map_revision=s['map_revision'],
                goal_age=0 if decision['terminate'] else s['age'], previous_goal_age=s['age'],
                decision=decision, reason=s['reason'], mode=executed['mode'], endpoint_arrived=endpoint_arrived,
                task_switch=bool(decision['terminate'] and np.linalg.norm(np.array(executed['endpoint'])-s['endpoint']) > 1e-4),
                position=s['position'].tolist(), velocity=s['velocity'].tolist(),
                event_target=target.tolist(),
                path_tracking_error=executed['cross_track'], distance_to_tracking=float(np.linalg.norm(s['position']-target)),
                estimated_route_seconds=s['estimated_seconds'], next_position=nxt[i]['position'].tolist(),
                executed=actual[i].tolist(), forced=s['forced'], emergency=s['emergency'],
                policy_executed=t['policy_executed'], reached=reached, collision=collision))
        row['path_option'] = self.transitions[-4:]
        return row

    def summary(self):
        r = RepairedMissionRuntime.summary(self)
        r.update(method=SCHEMA, task_network_trained=False,
            path_option=dict(trace=self.trace, options=self.o, variant='rule' if self.learner is None else self.learner.variant,
                intervention=self.intervention, original_task_contract=[28, 3, 124],
                safety_guarantee=False, learned_values_calibrated=False, shared_planning_context=True,
                strict_decentralization_claimed=False))
        return r
