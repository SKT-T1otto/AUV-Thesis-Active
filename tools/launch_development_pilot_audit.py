"""Reconcile the preregistered development pilot with its independent native branch audits."""
import argparse
from collections import Counter
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import median

import numpy as np

from core.env.target_motion import swept_relative_min_distance
from project_paths import ROOT
from tools.check_launch_candidates import write_receipt
from tools.launch_assignment_audit import audit_folder, require
from tools.launch_assignment_study import digest, file_hash, read


OUT = ROOT / 'verification/launch_development_pilot_v1'
REGISTRY = ROOT / 'verification/launch_assignment_v1/manifest.json'


def free_centers(scene):
    # Native M20 box. Count cell centers outside the closed physical obstacle boxes.
    grid = scene['planner_grid_size']
    axes = [(np.arange(n) + .5) * length / n for n, length in zip(grid, (20., 20., 8.))]
    centers = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1).reshape(-1, 3)
    for obstacle in scene['obstacles']:
        inside = np.all(np.abs(centers - obstacle['center']) <= .5 * np.asarray(obstacle['size']), axis=1)
        centers = centers[~inside]
    return centers


def coverage(branch, centers):
    """Retrospective geometry over actual detector checks. Never fed back to a boat."""
    radii = np.array(branch['protocol']['ranges']) + branch['protocol']['bias']
    covered = np.zeros(len(centers), dtype=bool)
    initial_covered = np.zeros_like(covered)
    checks = 0
    closest = None
    x = branch['initial']['target']['position']
    decisions = branch['episode']['decisions']
    next_draw = min((d['step'] for d in decisions if d['step'] > 0), default=branch['episode']['steps'])
    for row in branch['intervals']:
        e, y = row['event'], row['target']['position']
        if e['discovery_check_completed']:
            checks += 1
            for i, (start, end) in enumerate(zip(e['agent_start'][:3], e['agent_end'][:3])):
                delta = np.subtract(end, start)
                rel = centers - start
                divisor = np.dot(delta, delta)
                tau = np.clip(np.sum(rel * delta, axis=1) / divisor, 0., 1.) if divisor > 0 else np.zeros(len(rel))
                displacement = rel - tau[:, None] * delta
                reached = np.sum(displacement**2, axis=1) <= radii[i]**2
                covered |= reached
                if row['step'] <= next_draw:
                    initial_covered |= reached
                distance, _ = swept_relative_min_distance(start, end, x, y)
                candidate = (distance - radii[i], row['step'], i)
                if closest is None or candidate < closest:
                    closest = candidate
        x = y
    receipt = [d for d in decisions if 'departure' in d]
    return dict(detector_intervals=checks, first_continuation_request=next_draw if
                any(d['step'] > 0 for d in decisions) else None,
                continuation_requests=sum(d['step'] > 0 for d in decisions),
                free_grid_centers=len(centers), centers_until_first_continuation=int(initial_covered.sum()),
                centers_ever_covered=int(covered.sum()), closest_margin_m=float(closest[0]) if closest else None,
                closest_step=closest[1] if closest else None, closest_boat=closest[2] if closest else None,
                departure_routes=[d['route_id'] for d in receipt],
                departure_route_lengths_m=[sum(math.dist(a,b) for a,b in zip(points, points[1:]))
                    for d in receipt for points in [d['departure']['route_points']]])


def family_prefix(branch):
    pre = branch['prefix']
    return digest(dict(initial=branch['initial'], actor=branch['actor_sha256'], protocol=branch['protocol'],
                       menus=pre['menus'], draws=pre['draws'],
                       before=[s[:2] for s in pre['streams_before']], after=[s[:2] for s in pre['streams_after']]))


def departure_actions(branch):
    frame = branch['episode']['frames'][0]
    return digest(dict(commands=frame['commands'], roots=frame['roots'], routes=frame['routes'],
                       decisions=[d for d in branch['episode']['decisions'] if d['step'] == 0]))


def one_family(plan, family, scene, out):
    arm_names = plan['arms']
    counts = {name:Counter() for name in arm_names}
    contrast = {a+'-'+b: Counter(equal=0, first_only=0, second_only=0) for a,b in combinations(arm_names, 2)}
    reps, discordant = [], []
    signatures, first_actions, random_states = set(), {}, [set() for _ in range(3)]
    centers = free_centers(scene)
    frozen_prefix = None
    if not plan.get('reuse_archived_baseline', True):
        runtime = read(out / 'runtime.json')
        require(runtime['plan_sha256'] == file_hash(out / 'plan.json') and
                runtime['source_lock'] == plan['source_lock'], 'runtime receipt identity')
        saved, = [d for d in runtime['departures'] if d['scene'] == family['scene']]
        require(saved['family'] == family['family'] and
                family_prefix(saved['snapshot']) == saved['prefix_sha256'], 'frozen runtime departure identity')
        frozen_prefix = saved['prefix_sha256']
    for repeat in plan['repeats']:
        short = scene['scenario_id'].rsplit('_',1)[1]
        archived = plan.get('reuse_archived_baseline', True) and short == '0003' and repeat == 0
        folder = ROOT / plan['baseline'] if archived else out / f'{short}_r{repeat}'
        req = read(folder/'request.json')
        require((req['scene'],req['family'],req['repeat'],req['cap'],req['fixture'],req['order'],
                 req['source_lock'],req['actor_reference']['actor_parameters_sha256'],req['continuation'],req['terminal'])
                == (family['scene'],family['family'],repeat,plan['cap'],False,[0,1,2,3],plan['source_lock'],
                    plan['actor_sha256'],plan['continuation'],plan['terminal']), 'pilot cell mismatch')
        audited = audit_folder(folder)
        require(audited == read(folder/'audit.json'), 'stored audit differs')
        block = {}
        for arm in arm_names:
            b = read(folder/(arm+'.json'))
            prefix = family_prefix(b)
            if frozen_prefix is not None:
                require(prefix == frozen_prefix, 'branch differs from its frozen runtime departure')
            signatures.add(prefix)
            first = departure_actions(b)
            require(first_actions.setdefault(arm, first) == first, 'repeat changed the assigned departure')
            if arm == arm_names[0]:
                for seen, stream in zip(random_states,b['prefix']['streams_before']):
                    require(stream[2] not in seen, 'continuation stream reused across repeats')
                    seen.add(stream[2])
            l = audited['arms'][arm]['labels']
            require(not l['censored'] and l['S'] is not None, 'a censored arm cannot enter the terminal pilot')
            diag = coverage(b, centers)
            c = counts[arm]
            c.update(dict(n=1, found=int(l['ever_found']), received=int(l['ever_received']), success=int(l['S'])))
            block[arm] = dict(labels=l, coverage=diag, admissions=audited['arms'][arm]['admissions'])
        successes = [arm for arm in arm_names if block[arm]['labels']['S']]
        if 0 < len(successes) < len(arm_names):
            discordant.append(repeat)
        for a,b in combinations(arm_names,2):
            x,y = (block[n]['labels']['S'] for n in (a,b))
            contrast[a+'-'+b]['first_only' if x and not y else 'second_only' if y and not x else 'equal'] += 1
        reps.append(dict(repeat=repeat, folder=folder.relative_to(ROOT).as_posix(),
                         audit_sha256=file_hash(folder/'audit.json'), successful_arms=successes, arms=block))
    require(len(signatures)==1, 'different family prefix or departure candidate states across repeats')
    total = {k:sum(c[k] for c in counts.values()) for k in ('n','found','received','success')}
    coverage_rows = [r['arms'][a]['coverage'] for r in reps for a in arm_names]
    return dict(**family, counts=total, by_arm=counts, paired_success_contrasts=contrast,
                discordant_repeats=discordant, fixed_departure_digest=signatures.pop(),
                free_grid_centers=len(centers), median_centers_covered=median(d['centers_ever_covered'] for d in coverage_rows),
                median_initial_centers_covered=median(d['centers_until_first_continuation'] for d in coverage_rows),
                median_continuation_requests=median(d['continuation_requests'] for d in coverage_rows), blocks=reps)


def summarize(out):
    plan = read(out/'plan.json')
    require(file_hash(REGISTRY) == plan['registry_sha256'], 'different preregistered registry')
    for name, expected in plan['baseline_sha256'].items():
        require(file_hash(ROOT/plan['baseline']/name) == expected, 'repeat zero archive changed')
    registry = read(REGISTRY)
    order = []
    for fam in registry['families']:
        ids = sorted(m['scenario_id'] for m in fam['members'] if m['source']=='expansion24')
        if fam['split']=='development' and ids:
            order.append(dict(scene=ids[0],family=fam['family_id']))
    baseline_scene = read(ROOT/plan['baseline']/'request.json')['scene']
    selected = [f for f in order if f['scene']==baseline_scene] + [f for f in order if f['scene']!=baseline_scene][:2]
    require(order == plan['eligible_order'] and selected == plan['families'], 'different development selection')
    scenes = read((ROOT/registry['study']['native_inputs']['config']).with_name('scenes.json'))['scenarios']
    by_scene = {s['scenario_id']:s for s in scenes}
    results = [one_family(plan, f, by_scene[f['scene']],out) for f in selected]
    result = dict(schema='auv.launch_development_pilot.audit.v1', plan_sha256=file_hash(out/'plan.json'),
                  auditor_sha256=file_hash(Path(__file__)), families=results, **pilot_decision(results))
    if not plan.get('reuse_archived_baseline', True):
        result.update(schema='auv.launch_development_pilot.audit.v2',
                      runtime_receipt_sha256=file_hash(out / 'runtime.json'), archived_terminals_pooled=0,
                      historical_reference=dict(folder=plan['baseline'], sha256=plan['baseline_sha256']))
    return result


def pilot_decision(results):
    sparse = [f['scene'] for f in results if 2 * f['counts']['found'] < f['counts']['n']]
    discordant = {f['scene']: f['discordant_repeats'] for f in results if f['discordant_repeats']}
    return dict(low_discovery_families=sparse, inspect_search_coverage=len(sparse) >= 2,
                success_discordant_blocks=discordant, expand_paired_validation=bool(discordant))


def publish_summary(directory):
    result = summarize(directory)
    target = directory/'summary.json'
    if target.exists():
        require(read(target)==result, 'different saved pilot summary')
    else:
        write_receipt(target,result)
    for family in result['families']:
        print(family['scene'], family['counts'], family['discordant_repeats'])
    print(json.dumps({key: result[key] for key in ('expand_paired_validation', 'inspect_search_coverage')}))
    print(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUT)
    publish_summary(parser.parse_args().folder.resolve())
