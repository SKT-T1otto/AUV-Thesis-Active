from dataclasses import replace
from chapter3_bser.candidate_generator import generate_search_candidates
from chapter3_bser.candidate_generator import generate_standby_candidates
from chapter3_bser.online.allocator import BSEROnlineAllocator
import chapter3_bser.online.allocator as native_allocator
from chapter3_bser.online.types import OnlineAllocation
from chapter3_bser.types import CandidateGenerationResult
from core.mapping.travel_cost_service import TravelCostService
from chapter3_bser.safety.planning_refresh import start_status

class SearchPoolMixin:

    def search_pool(self, state, ids):
        state = self.owner.planning_views.search(state)
        ids = tuple(sorted(ids))
        generation = self.config['candidate_generation']
        if self.owner.safe_options['failure_policy'] and (not state.target_found):
            ready = []
            for agent_id in ids:
                reason = start_status(state, agent_id)
                if reason == 'ready':
                    ready.append(agent_id)
                else:
                    self.owner.safe_counts['start_preflight:' + reason] += 1
                    self.owner.safe_counts['short_circuited_candidate_pools'] += 1
            ids = tuple(ready)
        (candidates, unreachable, reasons) = generate_search_candidates(replace(state, searcher_ids=ids), TravelCostService(state), k_search=int(generation['k_search_exact']), minimum_separation=float(generation['minimum_separation']), maximum_travel_time=float(generation.get('maximum_physical_travel_time', generation.get('maximum_travel_time'))))
        self.owner.safe_counts['search_candidate_unreachable_queries'] += unreachable
        self.owner.safe_counts['candidate_shortage_events'] += sum((':ONLY_' in r for r in reasons))
        return (candidates, unreachable, reasons)

class SafeJointAllocator(SearchPoolMixin, BSEROnlineAllocator):

    def __init__(self, owner, config):
        super().__init__(config)
        self.owner = owner

    def _generate_candidates(self, state):
        (search, unreachable, reasons) = self.search_pool(state, state.searcher_ids)
        (standby, standby_unreachable, standby_reasons) = generate_standby_candidates(state, TravelCostService(state), k_standby=int(self.config['candidate_generation']['k_standby_exact']))
        return CandidateGenerationResult(search, standby, {i: sum((c.agent_id == i for c in search)) for i in state.searcher_ids}, unreachable, standby_unreachable, reasons + standby_reasons)

    def _partial_search_candidates(self, state, affected):
        return self.search_pool(state, affected)[0]

    def allocate_partial(self, state, current, *, affected_searcher_ids=(), executor_affected=False, trigger_reason):
        """Retain the native partial solve, including newly assigned searchers.

        The legacy merge iterates only previously assigned IDs. Use its exact
        candidate/solver/objective rules here when a requested searcher was
        absent, merging over old IDs union affected IDs. Ordinary partial
        replans continue through the unmodified inherited implementation.
        """
        affected = {int(value) for value in affected_searcher_ids}
        old_search = {item.agent_id: item for item in current.search_assignments}
        if not affected - old_search.keys():
            return super().allocate_partial(state, current, affected_searcher_ids=affected, executor_affected=executor_affected, trigger_reason=trigger_reason)
        if not affected.issubset(set(state.searcher_ids)):
            return (current, False, 'ATOMIC_REJECT_UNKNOWN_SEARCHER')
        candidates = self._partial_search_candidates(state, affected)
        observer = getattr(self, '_audit_candidate_generation_observer', None)
        if observer is not None:
            observer(state, candidates, None, scope='partial_search', agent_ids=tuple(sorted(affected)))
        if any((not any((c.agent_id == i for c in candidates)) for i in affected)):
            return (current, False, 'ATOMIC_REJECT_MISSING_SEARCH_ROUTE')
        frozen = tuple((self._frozen_search_candidate(item) for (i, item) in sorted(old_search.items()) if i not in affected))
        search = tuple(candidates) + frozen
        if executor_affected:
            (standby, _, _) = generate_standby_candidates(state, TravelCostService(state), k_standby=int(self.config['candidate_generation']['k_standby_exact']))
            if observer is not None:
                observer(state, (), standby, scope='partial_standby', agent_ids=())
        else:
            standby = (self._frozen_standby_candidate(current),)
        if not search or not standby:
            return (current, False, 'ATOMIC_REJECT_MISSING_LOCAL_CANDIDATES')
        context = native_allocator.build_objective_context(state, search, standby, self.config)
        solved = self._solve_candidates(search, standby, context)
        if solved.standby is None:
            return (current, False, 'ATOMIC_REJECT_MISSING_STANDBY_ROUTE')
        selected = {candidate.agent_id: candidate for candidate in solved.selected}
        if any((i not in selected for i in affected)):
            return (current, False, 'ATOMIC_REJECT_MISSING_GREEDY_SELECTION')
        frozen_by_id = {candidate.agent_id: candidate for candidate in frozen}
        (assignments, merged) = ([], [])
        for agent_id in sorted(old_search.keys() | affected):
            if agent_id in affected:
                candidate = selected[agent_id]
                assignments.append(self._search_assignment(candidate))
                merged.append(candidate)
            else:
                assignments.append(old_search[agent_id])
                merged.append(frozen_by_id[agent_id])
        if executor_affected:
            executor = self.execution.assign_standby(state, solved.standby)
            if not executor.reachable:
                return (current, False, 'ATOMIC_REJECT_EXECUTOR_UNREACHABLE')
            objective_standby = solved.standby
        else:
            executor = current.executor_assignment
            objective_standby = standby[0]
        objective = native_allocator.evaluate_objective(merged, objective_standby, context)
        diagnostics = native_allocator.response_diagnostics(merged, objective_standby, context)
        proposed = OnlineAllocation(search_assignments=tuple(assignments), executor_assignment=executor, objective_value=float(objective), detection_probability=float(native_allocator.expected_detection_probability(merged, context)), response_time=float(diagnostics.conditional_reachable_response_time), trigger_reason=trigger_reason, status=solved.status, search_frozen=False)
        return (proposed, True, 'ATOMIC_PARTIAL_BSER_PROPOSAL')
