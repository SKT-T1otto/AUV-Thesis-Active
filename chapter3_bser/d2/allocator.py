from dataclasses import replace
import math
from chapter3_bser.candidate_generator import generate_standby_candidates
from chapter3_bser.safety.allocators import SafeJointAllocator
from chapter3_bser.objective import build_objective_context
from chapter3_bser.objective import evaluate_objective
from chapter3_bser.objective import response_diagnostics
from chapter3_bser.online.types import OnlineAllocation
from core.mapping.travel_cost_service import TravelCostService
from chapter3_bser.d2.solver import solve
from chapter3_bser.d2.solver import fixed_greedy
from chapter3_bser.d2.solver import best_standby

def finite(value):
    return float(value) if math.isfinite(value) else None

class EffectAllocator(SafeJointAllocator):

    def __init__(self, owner, config, arm, anchor):
        super().__init__(owner, config)
        (self.arm, self.anchor) = (arm, tuple((float(x) for x in anchor)))
        self.audit = []
        self.last_standbys = ()
        self.standby_may_change = False

    @property
    def fixed_anchor(self):
        return self.arm in ('D0', 'D1')

    @property
    def pure(self):
        return self.arm in ('D0', 'D2')

    def anchor_assignment(self, state):
        item = self.execution._assignment(state, self.anchor, source='D_FIXED_INITIAL_ANCHOR')
        if not item.reachable:
            self.owner.effect_counts['anchor_unreachable'] += 1
        return item

    def standby_pool(self, state, current=None, *, movable=True):
        if not movable:
            return ((self._frozen_standby_candidate(current),), current.executor_assignment)
        if self.fixed_anchor:
            item = self.anchor_assignment(state)
            dummy = OnlineAllocation((), item, 0.0, 0.0, math.inf, 'anchor')
            return ((replace(self._frozen_standby_candidate(dummy), candidate_id='D_initial_anchor'),), item)
        (values, unreachable, reasons) = generate_standby_candidates(state, TravelCostService(state), k_standby=int(self.config['candidate_generation']['k_standby_exact']))
        self.owner.effect_counts['standby_unreachable_queries'] += unreachable
        self.owner.effect_counts['standby_shortage_events'] += sum((':ONLY_' in x for x in reasons))
        return (values, None)

    def _proposal(self, state, candidates, frozen, standbys, executor, trigger):
        self.last_standbys = tuple(standbys)
        self.standby_may_change = executor is None
        pool = tuple(candidates) + tuple(frozen)
        context = build_objective_context(state, pool, standbys, self.config)
        result = solve(self.arm, candidates, standbys, context, frozen=frozen)
        allocation = OnlineAllocation(tuple((self._search_assignment(c) for c in result.selected)), executor or self.execution.assign_standby(state, result.standby), result.objective, evaluate_objective(result.selected, result.standby, context, search_only=True), response_diagnostics(result.selected, result.standby, context).conditional_reachable_response_time, trigger, result.status)
        pure = fixed_greedy(candidates, standbys[0], context, pure=True, frozen=frozen)
        sequential_y = best_standby(pure.selected, standbys, context)
        selected_ids = [c.candidate_id for c in result.selected]
        pure_ids = [c.candidate_id for c in pure.selected]
        self.audit.append(dict(kind='proposal', step=int(state.step), trigger=trigger, candidates_by_agent={str(i): sum((c.agent_id == i for c in candidates)) for i in state.searcher_ids}, candidate_ids=[c.candidate_id for c in candidates], frozen_ids=[c.candidate_id for c in frozen], standby_count=len(standbys), selected_ids=selected_ids, pure_search_ids=pure_ids, search_differs_from_pure=selected_ids != pure_ids, standby=list(result.standby.waypoint), executor_reachable=allocation.executor_assignment.reachable, executor_route_time=finite(allocation.executor_assignment.estimated_arrival_time), objective=allocation.objective_value, detection=allocation.detection_probability, joint_score=evaluate_objective(result.selected, result.standby, context), sequential_joint_score=evaluate_objective(pure.selected, sequential_y, context), response_time=finite(allocation.response_time), installed=False))
        observer = getattr(self, '_audit_candidate_generation_observer', None)
        if observer is not None:
            observer(state, candidates, standbys, scope='D_proposal', agent_ids=tuple((c.agent_id for c in candidates)))
        return allocation

    def allocate(self, state, *, trigger_reason='online'):
        if state.target_found:
            return super().allocate(state, trigger_reason=trigger_reason)
        candidates = self.search_pool(state, state.searcher_ids)[0]
        (ys, executor) = self.standby_pool(state)
        if not ys:
            executor = self.anchor_assignment(state)
            dummy = OnlineAllocation((), executor, 0.0, 0.0, math.inf, trigger_reason)
            ys = (self._frozen_standby_candidate(dummy),)
            self.owner.effect_counts['standby_pool_fallback'] += 1
        return self._proposal(state, candidates, (), ys, executor, trigger_reason)

    def allocate_partial(self, state, current, *, affected_searcher_ids=(), executor_affected=False, trigger_reason):
        if state.target_found:
            return super().allocate_partial(state, current, affected_searcher_ids=affected_searcher_ids, executor_affected=executor_affected, trigger_reason=trigger_reason)
        affected = set(affected_searcher_ids)
        if not affected.issubset(state.searcher_ids):
            return (current, False, 'ATOMIC_REJECT_UNKNOWN_SEARCHER')
        candidates = self.search_pool(state, affected)[0] if affected else ()
        if any((not any((c.agent_id == i for c in candidates)) for i in affected)):
            return (current, False, 'ATOMIC_REJECT_MISSING_SEARCH_ROUTE')
        frozen = tuple((self._frozen_search_candidate(c) for c in current.search_assignments if c.agent_id not in affected))
        (ys, executor) = self.standby_pool(state, current, movable=executor_affected)
        if not ys:
            return (current, False, 'ATOMIC_REJECT_MISSING_STANDBY_ROUTE')
        proposed = self._proposal(state, candidates, frozen, ys, executor, trigger_reason)
        if not affected.issubset((c.agent_id for c in proposed.search_assignments)):
            return (current, False, 'ATOMIC_REJECT_MISSING_GREEDY_SELECTION')
        return (proposed, True, 'D_ATOMIC_CONSTRAINED_PROPOSAL')

    def rescore(self, state, allocation, *, finalize=False):
        if state.target_found or allocation.search_frozen:
            return allocation
        search = tuple((self._frozen_search_candidate(c) for c in allocation.search_assignments))
        y = self._frozen_standby_candidate(allocation)
        ys = self.last_standbys if finalize and self.arm == 'D2' and self.standby_may_change else (y,)
        context = build_objective_context(state, search, ys, self.config)
        if len(ys) > 1 or ys[0] is not y:
            y = best_standby(search, ys, context)
            allocation = replace(allocation, executor_assignment=self.execution.assign_standby(state, y))
        value = evaluate_objective(search, y, context, search_only=self.pure)
        scored = replace(allocation, objective_value=value, detection_probability=evaluate_objective(search, y, context, search_only=True), response_time=response_diagnostics(search, y, context).conditional_reachable_response_time)
        if finalize:
            self.audit.append(dict(kind='stabilized', step=int(state.step), selected_ids=[c.candidate_id for c in search], standby=list(y.waypoint), score_before_recompute=allocation.objective_value, objective=value, detection=scored.detection_probability, joint_score=evaluate_objective(search, y, context), response_time=finite(scored.response_time), installed=False))
        return scored
