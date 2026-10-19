from __future__ import annotations
from typing import Optional
from typing import Sequence
from chapter3_bser.objective import ObjectiveContext
from chapter3_bser.objective import marginal_gain
from chapter3_bser.types import SearchCandidate
from chapter3_bser.types import SolverResult
from chapter3_bser.types import StandbyCandidate

def solve_fixed_standby_greedy(candidates: Sequence[SearchCandidate], standby: Optional[StandbyCandidate], context: ObjectiveContext, *, search_only: bool=False) -> SolverResult:
    selected = []
    used_agents = set()
    ordered = sorted(candidates, key=lambda candidate: candidate.key)
    while True:
        feasible = [candidate for candidate in ordered if candidate.agent_id not in used_agents]
        if not feasible:
            break
        ranked = sorted(((-marginal_gain(selected, candidate, standby, context, search_only=search_only), candidate.key, candidate) for candidate in feasible), key=lambda item: (item[0], item[1]))
        gain = -ranked[0][0]
        if gain <= 1e-15:
            break
        chosen = ranked[0][2]
        selected.append(chosen)
        used_agents.add(chosen.agent_id)
    selected.sort(key=lambda candidate: candidate.key)
    from chapter3_bser.objective import evaluate_objective
    value = evaluate_objective(selected, standby, context, search_only=search_only)
    return SolverResult('search_only_greedy' if search_only else 'fixed_y_standard_greedy', tuple(selected), standby, value)

def solve_joint_greedy(candidates: Sequence[SearchCandidate], standby_candidates: Sequence[StandbyCandidate], context: ObjectiveContext) -> SolverResult:
    best = None
    for standby in sorted(standby_candidates, key=lambda item: item.key):
        current = solve_fixed_standby_greedy(candidates, standby, context)
        key = (tuple((candidate.key for candidate in current.selected)), standby.key)
        best_key = None if best is None else (tuple((candidate.key for candidate in best.selected)), best.standby.key)
        if best is None or current.objective > best.objective + 1e-15 or (abs(current.objective - best.objective) <= 1e-15 and key < best_key):
            best = current
    if best is None:
        return SolverResult('bser_standard_greedy', (), None, 0.0, status='NO_STANDBY_CANDIDATE')
    return SolverResult('bser_standard_greedy', best.selected, best.standby, best.objective)
