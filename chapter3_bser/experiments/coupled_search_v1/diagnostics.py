"""Bounded observers of public computation and installed guidance.

No random draws, planner decisions, budgets, likelihoods or physical actions
are changed here. Installation means guidance was installed; execution counts
use the next public physical transition and do not imply causal task benefit.
"""
from collections import Counter, OrderedDict, deque
from functools import wraps
import hashlib
import json
import time
import numpy as np


class StageAudit:
    def __init__(self, capacity=1024, clock=None):
        self.clock = clock or time.perf_counter
        self.trace = deque(maxlen=capacity)
        self.current = None
        self.counts = Counter()
        self.phase_seconds = Counter()
        self.failure_reasons = Counter()
        self.first_failure = None

    def begin(self, step, operation):
        now = self.clock()
        self.current = dict(step=int(step), operation=operation, started=now,
                            mark=now, phase='entry', phases={}, exceptions=[])

    def stage(self, name):
        if self.current is None:
            return
        row = self.current; now = self.clock()
        row['phases'][row['phase']] = row['phases'].get(row['phase'], 0.) + now-row['mark']
        row['phase'] = name; row['mark'] = now

    def note_failure(self, error):
        if self.current is not None:
            row = self.current
            item = dict(phase=row['phase'], reason=str(error))
            if len(row['exceptions']) < 4:
                row['exceptions'].append(item)

    def finish(self, *, error=None, **work):
        row = self.current
        if row is None:
            return
        phase = row['phase']; self.stage('finished'); now = self.clock()
        failed_phase = next((e['phase'] for e in reversed(row['exceptions']) if e['reason'] == str(error)), phase)
        event = dict(step=row['step'], operation=row['operation'], seconds=now-row['started'],
                     completed=error is None, error=None if error is None else str(error),
                     failure_phase=None if error is None else failed_phase,
                     first_exception=next(iter(row['exceptions']), None),
                     exceptions=row['exceptions'], phases=row['phases'], **work)
        self.trace.append(event); self.counts['attempts'] += 1
        self.counts['completed' if error is None else 'failed'] += 1
        for name, seconds in event['phases'].items():
            self.phase_seconds[name] += seconds
        if error is not None:
            self.failure_reasons[str(error)] += 1
            if self.first_failure is None:
                self.first_failure = dict(event)
        self.current = None

    def summary(self):
        values = [row['seconds'] for row in self.trace]
        return dict(counts=dict(self.counts), failure_reasons=dict(self.failure_reasons),
                    phase_seconds=dict(self.phase_seconds), first_failure=self.first_failure,
                    retained_attempts=len(values), trace_truncated=self.counts['attempts'] > len(values),
                    retained_p95=None if not values else float(np.percentile(values, 95)),
                    retained_max=None if not values else max(values), trace=list(self.trace))


def observe_inference(function):
    @wraps(function)
    def wrapped(self, evidence):
        if self.stopped:
            return function(self, evidence)
        audit = self.inference_diagnostics
        audit.begin(evidence.physical_step, 'inference')
        thrown = None
        try:
            return function(self, evidence)
        except Exception as error:
            thrown = error; audit.note_failure(error)
            raise
        finally:
            audit.finish(error=thrown or (None if self.valid else self.error),
                         retained_history=len(self.history),
                         replay_work=getattr(self, 'last_replay_work', None))
    return wrapped


def observe_planning(operation):
    def decorate(function):
        @wraps(function)
        def wrapped(self, state, *args, **kwargs):
            if state.target_found or (operation == 'rescore' and args and args[0].search_frozen):
                return function(self, state, *args, **kwargs)
            audit = self.planning_diagnostics
            audit.begin(state.step, operation); thrown = None
            try:
                result = function(self, state, *args, **kwargs)
                if operation == 'proposal':
                    self.session.activity.proposed(result, 'coupled', state.step)
                return result
            except Exception as error:
                thrown = error; audit.note_failure(error)
                raise
            finally:
                # Rescore catches an error and delegates to D2; retain that
                # failed core attempt even though a fallback value is returned.
                caught = audit.current['exceptions'] if audit.current else []
                audit.finish(error=thrown or (caught[0]['reason'] if caught else None))
        return wrapped
    return decorate


def observe_allocation(function):
    @wraps(function)
    def wrapped(self, state, *args, **kwargs):
        result = function(self, state, *args, **kwargs)
        if not state.target_found and self.failed_step == state.step:
            allocation = result[0] if isinstance(result, tuple) else result
            if not isinstance(result, tuple) or result[1]:
                self.session.activity.proposed(allocation, 'd2', state.step)
        return result
    return wrapped


def route_key(agent, identity, path, final):
    payload = [int(agent), str(identity), np.asarray(path, float).reshape(-1, 3).tolist(),
               np.asarray(final, float).tolist()]
    return hashlib.sha256(json.dumps(payload, separators=(',', ':')).encode()).hexdigest()


class ActivityAudit:
    def __init__(self, capacity=1024):
        self.capacity = capacity; self.trace = deque(maxlen=capacity)
        self.pending = OrderedDict(); self.pending_step = None
        self.accepted = OrderedDict(); self.installed = {}
        self.counts = Counter(); self.reasons = Counter(); self.fallback_reasons = Counter()
        self.fallback_step_count = 0; self.last_fallback_step = None
        self.first_fallback = None; self.first_fallback_reasons = Counter()
        self.distance = Counter(); self.last_transition = -1

    def proposed(self, allocation, origin, step):
        if self.pending_step != step:
            self.pending.clear(); self.pending_step = step
        for item in allocation.search_assignments:
            key = route_key(item.agent_id, item.candidate_id, item.path, item.waypoint)
            self.pending[key] = dict(agent=item.agent_id, identity=item.candidate_id,
                                     origin=origin, key=key)
            while len(self.pending) > self.capacity:
                self.pending.popitem(last=False)

    def decision(self, step, result, target_found=False):
        self.reasons[result.decision_reason] += 1
        if result.replanned:
            self.counts['accepted_replans'] += 1
            self.counts[('execution' if target_found else 'search')+'_accepted_replans'] += 1
            self._accept(result.allocation, step)

    def _accept(self, allocation, step):
        if self.pending_step != step:
            return
        origins=set()
        for item in allocation.search_assignments:
            key = route_key(item.agent_id, item.candidate_id, item.path, item.waypoint)
            if key in self.pending:
                self.accepted[key] = self.pending[key]; self.accepted.move_to_end(key)
                origins.add(self.pending[key]['origin'])
        for origin in origins:self.counts['accepted_'+origin+'_proposals'] += 1
        while len(self.accepted) > self.capacity:
            self.accepted.popitem(last=False)

    def fallback(self, step, error):
        self.fallback_reasons[str(error)] += 1
        if self.last_fallback_step != int(step):
            self.fallback_step_count += 1; self.last_fallback_step = int(step)
            self.first_fallback_reasons[str(error)] += 1
        if self.first_fallback is None:
            self.first_fallback = dict(step=int(step), reason=str(error))

    def install(self, context, allocation=None):
        if int(context.step) == 0 and allocation is not None:
            self._accept(allocation, 0)
        current = {}
        for item in context.agent_assignments:
            if item.agent_id not in (0, 1, 2):
                continue
            key = route_key(item.agent_id, item.assignment_id, item.planned_path, item.final_waypoint)
            exact = self.accepted.get(key)
            related = next((r for r in reversed(self.accepted.values())
                            if r['agent'] == item.agent_id and r['identity'] == item.assignment_id), None)
            if item.hold_state or not item.reachable:
                source = 'hold'
            elif exact is not None:
                source = exact['origin']+'_exact'
            elif related is not None:
                source = related['origin']+'_modified'
            else:
                source = 'unknown'
            current[item.agent_id] = source
        self.installed = current

    def advance(self, observation, step):
        if step <= self.last_transition:
            return
        self.last_transition = step
        phase = 'execution' if observation.found_before else 'search'
        self.counts[phase+'_physical_steps'] += 1
        sources = {}
        for agent in range(3):
            source = self.installed.get(agent, 'unknown'); sources[str(agent)] = source
            self.counts[phase+'_agent_steps'] += 1
            if phase == 'search':
                self.counts[source+'_agent_steps'] += 1
                self.distance[source] += float(np.linalg.norm(
                    np.asarray(observation.agent_end[agent])-observation.agent_start[agent]))
        self.trace.append(dict(step=int(step), phase=phase, sources=sources))

    def summary(self):
        denominator = self.counts['search_agent_steps']
        exact = self.counts['coupled_exact_agent_steps']
        return dict(counts=dict(self.counts), decision_reasons=dict(self.reasons),
                    fallback_reasons=dict(self.fallback_reasons),
                    fallback_search_steps=self.fallback_step_count,
                    first_fallback_reasons=dict(self.first_fallback_reasons), first_fallback=self.first_fallback,
                    coupled_exact_agent_step_fraction=None if not denominator else exact/denominator,
                    movement_distance_by_source=dict(self.distance), trace=list(self.trace),
                    trace_truncated=sum(self.counts[k] for k in ('search_physical_steps', 'execution_physical_steps')) > len(self.trace),
                    scope='Public physical transitions under installed guidance; modified paths and holds are separate. Not causal action attribution.')
