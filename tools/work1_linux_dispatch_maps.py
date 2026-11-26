"""One native launch menu, one executable reference per boat/region, frozen map forecasts."""
from dataclasses import replace
from itertools import permutations

import numpy as np

from chapter3_bser.d2.solver import fixed_greedy
from chapter3_bser.detection_model import detection_probability
from chapter3_bser.experiments.coupled_search_v1.execution_forecast import PublicExecutionErrors
from chapter3_bser.objective import marginal_gain
from chapter3_bser.path_option.path_context import public_route
from chapter3_bser.predispatch_evidence import predict_public_offer
from chapter3_bser.sweep_events import detection_radii
from core.mapping.planning_state import extract_planning_state, planning_state_sha256

MODELS = ('history', 'bins')
NATIVE_NAMES = {'history': 'C', 'bins': 'new'}
H = 16
TOL = 1e-15


def earliest_step(position, center, radius, spec, dt):
    distance = max(0., np.linalg.norm(np.subtract(position, center)) - radius)
    step = dt * np.hypot(spec['v_xy_max'], spec['v_z_max'])
    return max(1, int(np.ceil(distance / step - 1e-12)))


def assignments(base, candidates, context, standby):
    """Preserve the native task multiset; keep only strictly positive greedy prefixes."""
    by_pair = {(c.agent_id, tuple(c.waypoint)): c for c in candidates}
    if len(by_pair) != len(candidates):
        raise ValueError('Ambiguous native boat/region route')
    options = {}
    goals = [tuple(c.waypoint) for c in base]
    available = sorted({c.agent_id for c in candidates})
    for owners in permutations(available, len(base)):
        keys = list(zip(owners, goals))
        if not all(key in by_pair for key in keys):
            continue
        remaining = sorted((by_pair[key] for key in keys), key=lambda c:c.key)
        plan = tuple(c.candidate_id for c in remaining)
        selected, priority = [], []
        while remaining:
            gain, key, index = min((-marginal_gain(selected, c, standby, context, search_only=True), c.key, j)
                                   for j, c in enumerate(remaining))
            if -gain <= TOL:
                break
            priority.append((gain, key))
            selected.append(remaining.pop(index))
        if not remaining:
            options[plan] = priority
    anchor = tuple(c.candidate_id for c in base)
    return sorted(options, key=lambda p: (p != anchor, options[p]))


def forecast_offer(state, c, query, history, model, specs, contract, radii, clearance, arrival):
    i = c.agent_id
    cell = int(np.argmin(np.linalg.norm(state.grid.cell_centers - c.waypoint, axis=1)))
    if not np.allclose(state.grid.cell_centers[cell], c.waypoint, atol=1e-5, rtol=0):
        raise ValueError('Task is not a native spatial cell center')
    preview, evidence = predict_public_offer(state, i, c.waypoint, query, tokens=[c.candidate_id],
        prior_receipt=None, history_scenarios=history, calibration=model, specs=specs, contract=contract,
        clearance=clearance, arrival_radius=arrival, sensor_radius=radii[i])
    record = dict(boat=i, cell=cell, goal=c.waypoint, support=evidence.route_support,
                  same_vertices=np.array_equal(c.path_points, query.path_points),
                  initial_footprint=bool(cell in evidence.first_scan_cells),
                  lower_step=earliest_step(state.agents[i].position, c.waypoint, radii[i], specs[i], contract['dt']),
                  route_id=preview.route_id if preview else None, probabilities={})
    draws = {}
    if preview is not None:
        for m, name in NATIVE_NAMES.items():
            first = preview.first[name]
            low = (first > 0) & (first <= H)
            low[:, evidence.first_scan_cells] = True
            high = low.copy()
            high[np.asarray(preview.valid_steps[name]) < H] = True
            draws[m] = (low, high, preview.weights)
            record['probabilities'][m] = [float(preview.weights @ a[:, cell]) for a in (low, high)]
    return record, draws


def metrics(plan, offers, samples, belief, m):
    if any(m not in samples[k] for k in plan):
        return None
    parts = [samples[k][m] for k in plan]
    w = parts[0][2]
    if any(not np.array_equal(w, part[2]) for part in parts):
        raise ValueError('Joint execution sample weights differ')
    out = dict(region=[], coverage=[], boats=[])
    for bound in (0, 1):
        hits = np.stack([p[bound] for p in parts])
        out['boats'].append(((hits @ belief) @ w).tolist())
        out['region'].append(sum(float(w @ a[:, offers[k]['cell']]) for k, a in zip(plan, hits)))
        out['coverage'].append(float(w @ (hits.any(axis=0) @ belief)))
    return out


def choose(scores, key):
    """Spatial wins ties or unresolved comparisons, including an unknown competitor."""
    if not scores or any(s is None for s in scores):
        return 0
    intervals = [s[key] for s in scores]
    for j, (lo, hi) in enumerate(intervals[1:], 1):
        if lo > intervals[0][1] + TOL and all(j == k or lo >= v[1] - TOL for k, v in enumerate(intervals)):
            return j
    return 0


def decide_branch(plans, offers, choices, ordinal):
    """One allocation fork per family, chosen without outcomes."""
    if len(plans) <= 1:
        return None, 'single_assignment'
    models = MODELS if ordinal % 2 == 0 else MODELS[::-1]
    preferred = [choices[m][key] for key in ('region', 'coverage') for m in models]
    valid = [j for j, p in enumerate(plans) if j and
             all(offers[k]['support'] == 'route_available' for k in set(p) | set(plans[0]))]
    def tests_two_boats(j):
        a, b = set(plans[0]), set(plans[j])
        return any(offers[x]['cell'] == offers[y]['cell'] and offers[x]['boat'] != offers[y]['boat']
                   and max(offers[x]['lower_step'], offers[y]['lower_step']) <= H for x in a-b for y in b-a)
    for pool, label in (([j for j in valid if tests_two_boats(j)], 'competition'), (valid, 'coverage')):
        if pool:
            index = next((j for j in preferred if j in pool), pool[0])
            return index, label + ('_policy' if index in preferred else '_diagnostic')
    return None, 'no_supported_alternative'


def snapshot(allocator, state, candidates, result, context, model, scenario_config, arrival, ordinal):
    owner = allocator.owner
    live = extract_planning_state(owner.env)
    if live.step != 0 or state.step != 0 or not np.array_equal(context.belief, live.target_belief.probabilities):
        raise ValueError('Launch clock or public belief mismatch')
    specs = owner.env.get_public_vehicle_specs()
    control = owner.env.get_public_control_contract()
    contract = dict(control, **owner._original_composition)
    history = PublicExecutionErrors(scenario_config, specs, contract).scenarios(0)
    radii = detection_radii(specs, control['detect_eps_bias'])
    offers, samples, kernels = {}, {}, {}
    for c in candidates:
        query = public_route(owner.env, c.agent_id, c.waypoint)
        offers[c.candidate_id], samples[c.candidate_id] = forecast_offer(
            live, c, query, history, model, specs, contract, radii, owner.planning_views.clearance, arrival)
        if query.reachable:
            common = replace(c, path_points=query.path_points)
            kernels[c.candidate_id] = detection_probability(common, state, allocator.config)
    base = tuple(sorted(result.selected, key=lambda c: c.key))
    anchor = tuple(c.candidate_id for c in base)
    prior = fixed_greedy(candidates, result.standby, context, pure=True)
    if tuple(c.candidate_id for c in prior.selected) != anchor:
        raise ValueError('Disabled maps must reproduce native D2')
    native_plans = assignments(base, candidates, context, result.standby)
    plans = native_plans
    gate = 'passed'
    if [c.agent_id for c in base] != [0, 1, 2]:
        gate = 'incomplete_team'
    elif len(kernels) != len(candidates):
        gate = 'missing_reference'
    else:
        shared = replace(context, detection_by_id=kernels)
        common = fixed_greedy(candidates, result.standby, shared, pure=True)
        if tuple(c.candidate_id for c in common.selected) != anchor:
            gate = 'neutral_assignment_changed'
        else:
            shared_plans = assignments(base, candidates, shared, result.standby)
            plans = [p for p in shared_plans if p in native_plans]
    if not plans or plans[0] != anchor:
        raise ValueError('Native anchor missing')
    scores, choices = {}, {}
    for m in MODELS:
        scores[m] = [metrics(p, offers, samples, context.belief, m) if len(p) == 3 else None for p in plans]
        choices[m] = {k: choose(scores[m], k) if gate == 'passed' else 0 for k in ('region', 'coverage')}
    branch, reason = (decide_branch(plans, offers, choices, ordinal) if gate == 'passed'
                      else (None, gate))
    cells = sorted({offers[k]['cell'] for k in anchor})
    eligibility = {}
    def exchanges(cell, pool):
        owners = [{offers[k]['boat'] for k in p if offers[k]['cell'] == cell} for p in pool]
        return sorted({tuple(sorted((i,j))) for left in owners for right in owners
                       for i in left-right for j in right-left})
    for cell in cells:
        options = {k for p in native_plans for k in p if offers[k]['cell'] == cell}
        eligible = {k for p in plans for k in p if offers[k]['cell'] == cell}
        native_pairs = exchanges(cell,native_plans)
        supported_plans = [p for p in plans if all(offers[k]['support']=='route_available' for k in p)]
        bounds = {offers[k]['boat']:offers[k]['lower_step'] for k in options}
        eligibility[cell] = dict(native_pairs=native_pairs,
            possible_pairs=[pair for pair in native_pairs if max(bounds[i] for i in pair)<=H],
            comparable_pairs=[pair for pair in exchanges(cell,supported_plans) if gate=='passed'
                              and max(bounds[i] for i in pair)<=H],
            menu=sorted({a['boat'] for a in offers.values() if a['cell']==cell}),
            native=sorted(offers[k]['boat'] for k in options),
            possible=sorted(offers[k]['boat'] for k in options if offers[k]['lower_step'] <= H),
            comparable=sorted(offers[k]['boat'] for k in eligible if gate=='passed' and
                              offers[k]['lower_step'] <= H and offers[k]['support']=='route_available'))
    return dict(state_digest=planning_state_sha256(live), gate=gate, offers=offers, plans=plans,
                scores=scores, choices=choices, branch=branch, branch_reason=reason, eligibility=eligibility,
                standby=dict(id=result.standby.candidate_id, goal=result.standby.waypoint),
                belief=context.belief, radii=radii, centers=live.grid.cell_centers)
