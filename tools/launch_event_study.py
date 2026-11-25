"""Seal step-four development predictions from reset; score independently audited terminals."""
import argparse
from math import log, sqrt

import numpy as np
import torch

from chapter3_bser.handoff_value.event_model import Window, contrast_bounds, event_rows, standby_contrasts
from project_paths import ROOT
from tools.check_launch_candidates import write_receipt
from tools.launch_assignment_audit import require
from tools.launch_assignment_study import digest, file_hash, read
from tools.launch_development_pilot_audit import summarize
from tools.launch_intervention_source import checked_source
from tools.run_launch_development_pilot import LOCAL_OUT, preview


STUDY = ROOT / 'verification/launch_event_model_v1'
PILOT_HASH = 'b58863102d4f50cf5010e90fe2b734f59ca30eeb7af51f34232decf9946267d5'
CODE = ('chapter3_bser/handoff_value/event_model.py', 'tools/launch_event_study.py')


def predict(menus, standby, protocol):
    require([m['boat'] for m in menus] == [0, 1, 2] and all(m['step'] == 0 for m in menus),
            'expected native step-zero menu order')
    points = [m['points'] for m in menus]
    if any(len(p) < n for p, n in zip(points, (2, 2, 1))):
        return dict(status='four_way_assignment_unavailable')
    # Projected goals do not guarantee admission, detection or receipt.
    windows = [[Window(1) for _ in range(n)] for n in (2, 2, 1)]
    rows = event_rows(windows, {}, horizon=protocol['horizon'], hold=protocol['hold'])
    radii = np.asarray(protocol['ranges']) + protocol['bias']
    proxy = standby_contrasts(points, standby, radii)
    result = {}
    for b in (0, 1):
        contrast = {(1,b,0): 1, (0,b,0): -1}
        lo, hi = contrast_bounds([rows], contrast)
        envelope = proxy[str(b)]
        signed_count = envelope['constant_value']
        result[str(b)] = dict(native_bounds=[int(lo), int(hi)], proxy=envelope,
                              predicted_sign=(signed_count > 0) - (signed_count < 0))
    return dict(status='development_prediction', contrasts=result)


def identity():
    require(file_hash(LOCAL_OUT/'plan.json') == PILOT_HASH, 'different pilot plan')
    plan = read(LOCAL_OUT/'plan.json')
    require(checked_source() == plan['source_lock'], 'different native source')
    require([x['scene'].rsplit('_', 1)[-1] for x in plan['families']] == ['0003','0020','0014'],
            'unexpected family order')
    return plan, dict(plan_sha256=PILOT_HASH, source_sha256={p: file_hash(ROOT/p) for p in CODE})


def prediction(departure):
    snapshot = departure['snapshot']
    return predict(snapshot['prefix']['menus'], snapshot['initial']['standby'], snapshot['protocol'])


def save_once(path, record):
    if path.exists():
        require(read(path) == record, 'conflicting receipt: ' + str(path))
    else:
        write_receipt(path, record)


def seal():
    plan, provenance = identity()
    torch.set_num_threads(1)
    # preview() only opens the historical branch for the first (excluded) family.
    predictions = {}
    for family in plan['families'][1:]:
        dep = preview(plan, family)
        predictions[family['scene']] = dict(family=family['family'], prediction=prediction(dep))
    record = dict(schema='auv.launch_events.seal.v1', **provenance, predictions=predictions,
                  axes='S(1,b,0)-S(0,b,0), b=0,1', alpha=.05, repeats=plan['repeats'])
    save_once(STUDY/'predictions.json', record)
    return record


def assess(values, sign, comparisons=4):
    """Paired continuation differences within one family, fixed-look simultaneous interval."""
    if not values or any(v not in (-1,0,1) for v in values) or sign not in (-1, 0, 1):
        raise ValueError('invalid paired contrast')
    mean = sum(values) / len(values)
    width = sqrt(2 * log(2 * comparisons / .05) / len(values))
    low, high = max(-1., mean-width), min(1., mean+width)
    empirical = (mean > 0) - (mean < 0)
    supported = 1 if low > 0 else -1 if high < 0 else 0
    verdict = 'unresolved'
    if sign and supported:
        verdict = 'supported' if supported == sign else 'contradicted'
    return dict(paired=values, estimate=mean, empirical_sign=empirical,
                simultaneous_interval=[low,high], verdict=verdict)


def score():
    plan, provenance = identity()
    seal_path = STUDY/'predictions.json'
    sealed = read(seal_path)
    require(all(sealed[k] == v for k,v in provenance.items()), 'seal/source mismatch')
    targets = plan['families'][1:]
    require(set(sealed['predictions']) == {f['scene'] for f in targets} and
            sealed['repeats'] == plan['repeats'] and sealed['alpha'] == .05, 'different sealed design')
    # Analyze only complete, independently reconstructed pilot receipts.
    summary = summarize(LOCAL_OUT)
    require(summary == read(LOCAL_OUT/'summary.json') and summary['archived_terminals_pooled'] == 0,
            'different saved v2 terminal summary')
    departures = {d['scene']:d for d in read(LOCAL_OUT/'runtime.json')['departures']}
    families = {f['scene']:f for f in summary['families']}
    evaluated = {}
    for f in targets:
        scene = f['scene']
        spec = sealed['predictions'][scene]
        require(spec['family'] == families[scene]['family'] == f['family'], 'changed family')
        local = prediction(departures[scene])
        if local != spec['prediction'] or local['status'] != 'development_prediction':
            evaluated[scene] = dict(status='runtime_prediction_differs_or_unavailable', local=local)
            continue
        blocks = families[scene]['blocks']
        require([r['repeat'] for r in blocks] == plan['repeats'], 'changed repeats')
        comparisons = {}
        for b in ('0','1'):
            outcomes = [[r['arms'][a+b]['labels']['S'] for a in ('0','1')] for r in blocks]
            require(all(type(s) is bool for pair in outcomes for s in pair), 'unknown terminal')
            comparisons[b] = assess([int(y)-int(x) for x,y in outcomes], local['contrasts'][b]['predicted_sign'])
        evaluated[scene] = dict(status='scored', contrasts=comparisons)
    record = dict(schema='auv.launch_events.score.v1', seal_sha256=file_hash(seal_path),
                  summary_digest=digest(summary), evaluated=evaluated)
    save_once(STUDY/'pilot_score.json', record)
    return record


if __name__ == '__main__':
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('seal','score'))
    args = parser.parse_args()
    print(json.dumps((seal if args.command == 'seal' else score)(), indent=2))
