from chapter3_bser.objective import evaluate_objective
from chapter3_bser.objective import marginal_gain
from chapter3_bser.types import SolverResult

def fixed_greedy(candidates, standby, context, *, pure=False, frozen=()):
    selected = list(frozen)
    used = {c.agent_id for c in selected}
    if len(used) != len(selected):
        raise ValueError('duplicate frozen searcher')
    while True:
        feasible = [c for c in candidates if c.agent_id not in used]
        if not feasible:
            break
        ranked = sorted(((-marginal_gain(selected, c, standby, context, search_only=pure), c.key, c) for c in feasible), key=lambda x: (x[0], x[1]))
        if -ranked[0][0] <= 1e-15:
            break
        selected.append(ranked[0][2])
        used.add(ranked[0][2].agent_id)
    selected = tuple(sorted(selected, key=lambda c: c.key))
    return SolverResult('controlled_greedy', selected, standby, evaluate_objective(selected, standby, context, search_only=pure))

def best_standby(selected, standbys, context):
    (best, value) = (None, -1.0)
    for y in sorted(standbys, key=lambda c: c.key):
        score = evaluate_objective(selected, y, context)
        if score > value + 1e-15:
            (best, value) = (y, score)
    return best

def solve(arm, candidates, standbys, context, *, frozen=()):
    if arm not in ('D0', 'D1', 'D2', 'D3') or not standbys:
        raise ValueError('D solve requires an arm and a nonempty standby pool')
    ys = tuple(sorted(standbys, key=lambda c: c.key))
    if arm in ('D0', 'D1') and len(ys) != 1:
        raise ValueError('fixed-anchor arms require exactly one standby')
    if arm in ('D0', 'D2'):
        result = fixed_greedy(candidates, ys[0], context, pure=True, frozen=frozen)
        y = ys[0] if arm == 'D0' else best_standby(result.selected, ys, context)
        return SolverResult(arm, result.selected, y, result.objective)
    best = None
    for y in ys:
        result = fixed_greedy(candidates, y, context, frozen=frozen)
        key = (tuple((c.key for c in result.selected)), y.key)
        best_key = None if best is None else (tuple((c.key for c in best.selected)), best.standby.key)
        if best is None or result.objective > best.objective + 1e-15 or (abs(result.objective - best.objective) <= 1e-15 and key < best_key):
            best = result
    return SolverResult(arm, best.selected, best.standby, best.objective)
