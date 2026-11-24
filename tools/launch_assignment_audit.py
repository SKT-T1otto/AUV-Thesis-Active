"""Read-only audit of saved joint departures. No actor, mission or collector is imported."""
import argparse
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from core.env.target_motion import TargetState, simulate_target_trajectory, swept_relative_min_distance
from project_paths import ROOT


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def labels(initial, intervals, protocol, cap):
    """Rebuild registered discovery, physical packet receipt and native terminal S."""
    require(0 < len(intervals) <= cap <= protocol['horizon'], 'invalid observed interval count')
    prev = initial['kinematics']['positions']
    x = initial['target']['position']
    found = finder = received = success_step = None
    hits, hold, rewards = [], 0, []
    for step, row in enumerate(intervals, 1):
        e = row['event']
        p, q, y = e['agent_start'], e['agent_end'], row['target']['position']
        require(row['step'] == step and prev == p and e['found_before'] == (found is not None), 'interval prefix')
        collision = row['collision']
        require(e['target_transition_completed'] == (not collision) and
                e['discovery_check_completed'] == (found is None and not collision), 'native detector clock')
        if e['discovery_check_completed']:
            candidates = [(swept_relative_min_distance(p[i], q[i], x, y)[0], i) for i in range(3)]
            eligible = [(d, i) for d, i in candidates if d <= protocol['ranges'][i] + protocol['bias']]
            if eligible:
                found, finder = step, min(eligible)[1]
                hits = [i for _, i in eligible]
        require(e['found_after'] == (found is not None) and row['finder'] == (-1 if finder is None else finder),
                'native registration disagrees with geometry')
        packet = row['packet']
        due = found is not None and step > found
        require((packet is not None) == due, 'missing or premature public receipt')
        if packet is not None:
            require(packet['physical_step'] == step and packet['received_step'] == found + 1 and
                    packet['payload']['sample_step'] == found, 'packet clocks')
            if received is None:
                received = step
                for k in ('position', 'velocity'):
                    require(packet['payload'][k] == intervals[found - 1]['target'][k], 'packet sample')
        if packet is not None and not collision:
            inside = max(math.dist(p[3], x), math.dist(q[3], y)) <= protocol['radius']
            hold = hold + 1 if inside else 0
            if hold == protocol['hold']:
                success_step = step
        require(row['native_success'] == (success_step is not None), 'native success disagrees with hold')
        reason = ('obstacle_collision' if collision else 'success' if success_step is not None else
                  'timeout' if step == protocol['horizon'] else 'running')
        ended = reason != 'running'
        require(e['terminal_reason'] == reason and e['episode_terminated'] == ended and
                (not ended or step == len(intervals)), 'terminal clock or reason')
        require(len(row['source_rewards']) == 4 and row['broadcast'] == [row['team']] * 4 and
                math.isclose(np.mean(row['source_rewards'], dtype=np.float32), row['team'], abs_tol=1e-6),
                'team reward accounting')
        rewards.append(row['team'])
        x, prev = y, q
    require(ended or len(intervals) == cap, 'premature observation stop')
    g = math.fsum((.95 ** t) * reward for t, reward in enumerate(rewards))
    return dict(steps=len(rewards), censored=not ended, S=(success_step is not None) if ended else None,
                reason=reason if ended else 'observation_cap', found_step=found, registered_finder=finder,
                ever_found=True if found is not None else False if ended else None,
                ever_received=True if received is not None else False if ended else None,
                same_step_hits=hits, receipt_step=received, success_step=success_step,
                observed_discounted_return=g, terminal_discounted_return=g if ended else None)


def admissions(branch):
    ep, pre = branch['episode'], branch['prefix']
    decisions = [d for d in ep['decisions'] if 'departure' in d]
    require([d['agent'] for d in decisions] == [0, 1, 2] and all(d['step'] == 0 for d in decisions),
            'expected one departure per search boat')
    require(len(pre['menus']) == len(pre['draws']) == len(pre['streams_before']) == len(pre['streams_after']) == 3,
            'incomplete departure prefix')
    output = []
    reports = {r['origin']: r for r in ep['communication']['reports'] if r['step'] == 0}
    for i, d in enumerate(decisions):
        menu, attempts, receipt = pre['menus'][i], pre['draws'][i], d['departure']
        require(menu['boat'] == i and menu['step'] == 0 and menu['features'] == d['features'] and
                attempts == receipt['draws'] and [a['stream'] for a in attempts] == ['original_offer','extra_offer'],
                'addressed menu or draw identity')
        a, b = pre['streams_before'][i], pre['streams_after'][i]
        require(len(a) == len(b) == 3 and a[2] == b[2] and a[0] != b[0] and a[1] != b[1], 'private streams')
        points = []
        for draw in attempts:
            point = draw['projected']
            status = 'infeasible' if point is None else 'duplicate' if point in points else 'feasible'
            if status == 'feasible':
                points.append(point)
            require(draw['status'] == status and draw['option'] == (points.index(point) if point is not None else None),
                    'effective projection identity')
        require(points == menu['points'] == receipt['options'], 'projected menu')
        actual = branch['assignment'][i]
        chosen = next((a for a in attempts if a['status'] == 'feasible' and a['option'] == actual), None)
        require((chosen is not None or (not points and actual is None)) and receipt['selected_option'] == actual and
                receipt['selected_stream'] == (chosen['stream'] if chosen else None) and
                d['proposed'] == (chosen['projected'] if chosen else None), 'assigned versus submitted')
        frame = ep['frames'][0]
        require(d['route_id'] == frame['routes'][i] and d['root'] == frame['roots'][i], 'installed task or route')
        installed = bool(chosen and receipt['task_id'] > 0 and receipt['admission_status'] not in ('RELEASED','CONNECTOR')
                         and np.allclose(d['root'], chosen['projected'], atol=1e-6, rtol=0))
        reached = bool(installed and receipt['admission_status'] == 'ACTIVE' and receipt['route_reachable'] and
                       receipt['route_points'] and np.allclose(receipt['route_points'][-1], chosen['projected'], atol=1e-6, rtol=0))
        require(installed == d['accepted'] and reached == receipt['route_to_selected'], 'acceptance or route')
        report = reports[i]
        require(report['task_id'] == receipt['task_id'] and
                report['root'] == (frame['roots'][i] if report['active'] else None) and
                (not installed or report['active']), 'independently reported native task')
        if reached:
            require(report['remaining_route'] and np.allclose(report['remaining_route'][-1], chosen['projected'],
                    rtol=0, atol=1e-6), 'executed route differs from reported route')
        fallback = ('no_feasible_offer' if not chosen else 'native_rejection' if not installed else
                    'native_route_differs_or_unreachable' if not reached else None)
        require(receipt['fallback'] == fallback, 'fallback coding')
        output.append(dict(agent=i, assigned=actual, installed_root=d['root'], task_id=receipt['task_id'],
                           accepted=installed, route_id=d['route_id'], follows=reached, fallback=fallback))
    return output


def check_reset(branch, req, study, scene):
    import torch
    init, protocol = branch['initial'], branch['protocol']
    require(init['scene_digest'] == digest(scene) and init['task']['step'] == 0 and
            init['prior_task_ids'] == [0]*4 and init['inboxes'] == [[], [], [], []] and init['packet'] is None and
            init['standby'] == scene['initial_executor_wait_point'], 'reset or standby mismatch')
    require(np.array_equal(init['kinematics']['positions'], np.asarray(scene['initial_agent_positions'], np.float32)),
            'physical launch positions')
    require(protocol == dict(ranges=np.asarray([2., 2.35, 2.75], np.float32).tolist(), bias=.1,
                             radius=scene['target_capture_radius'], hold=5, horizon=scene['max_steps']), 'native protocol')
    for i, states in enumerate(branch['prefix']['streams_before']):
        for index, kind in enumerate(('original_offer','extra_offer','continuation')):
            repeat = req['repeat'] if index == 2 else 0
            seed = int(digest([study['random_streams']['master_seed'], req['family'], kind, i, repeat])[:16], 16)
            g = torch.Generator().manual_seed(seed)
            expected = hashlib.sha256(g.get_state().numpy().tobytes()).hexdigest()
            require(expected == states[index], 'unpaired seed')
            if index != 2:
                torch.randn(3, generator=g)
            expected = hashlib.sha256(g.get_state().numpy().tobytes()).hexdigest()
            require(expected == branch['prefix']['streams_after'][i][index], 'unexpected draw count')
    path = simulate_target_trajectory(TargetState(np.asarray(scene['target_initial_position'], np.float32),
        scene['target_initial_velocity'], 0, scene['target_motion_mode']), len(branch['intervals']), .2,
        (20.,20.,8.), scene['obstacles'], clearance=scene['target_obstacle_clearance'])
    require(np.allclose(init['target']['position'], path[0].position, atol=1e-6, rtol=0), 'target reset')
    for i, row in enumerate(branch['intervals'], 1):
        expected = path[i-1 if row['collision'] else i]
        for key in ('position','velocity'):
            require(np.allclose(row['target'][key], getattr(expected, key), rtol=0, atol=1e-6), 'target trajectory')


def audit_folder(folder):
    load = lambda name: json.loads((folder / name).read_bytes())
    req = load('request.json')
    require(req['schema'] == 'auv.launch_intervention.v1' and req['split'] == 'development' and
            req['gamma'] == .95 and req['assignment_axes'] == [0, 1] and req['third_option'] == 0, 'study design')
    registry = json.loads((ROOT / 'verification/launch_assignment_v1/manifest.json').read_bytes())
    study = registry['study']
    lock_name = 'provenance/launch_interventions_v1.json'
    source_hashes = {hashlib.sha256((ROOT / lock_name).read_bytes()).hexdigest()}
    first_archive = ROOT / 'provenance/launch_intervention_first_collection.zip'
    if first_archive.exists():
        with ZipFile(first_archive) as saved:
            source_hashes.add(hashlib.sha256(saved.read(lock_name)).hexdigest())
    require(req['actor_reference'] == study['candidates']['proposer_reference'] and
            sorted(req['order']) == list(range(4)) and req['continuation'] == 'unchanged_neighbor_policy' and
            req['terminal'] == 'native_direct' and req['source_lock'] in source_hashes, 'protocol identity')
    scenes = json.loads((ROOT / study['native_inputs']['config']).with_name('scenes.json').read_bytes())['scenarios']
    scene, = [s for s in scenes if s['scenario_id'] == req['scene']]
    require(any(f['family_id'] == req['family'] and f['split'] == 'development' and
                any(m['physical'] == req['physical'] and m['scenario_id'] == req['scene'] for m in f['members'])
                for f in registry['families']), 'unregistered development family')
    prefix = None
    rows, hashes = {}, {}
    for arm in ('00', '01', '10', '11'):
        name = arm + '.json'
        branch = load(name)
        hashes[arm] = hashlib.sha256((folder / name).read_bytes()).hexdigest()
        require(branch['arm'] == arm and branch['request_digest'] == digest(req) and
                branch['assignment'] == [int(arm[0]), int(arm[1]), 0], 'branch assignment or request')
        evidence = [branch[k] for k in ('initial','prefix','protocol','actor_sha256')]
        if prefix is None:
            prefix = evidence
        require(evidence == prefix, 'branches do not share the predecision state, offers and random streams')
        check_reset(branch, req, study, scene)
        if not req['fixture']:
            require(branch['actor_sha256'] == req['actor_reference']['actor_parameters_sha256'], 'frozen actor')
        ep = branch['episode']
        outcome = labels(branch['initial'], branch['intervals'], branch['protocol'], req['cap'])
        require((ep['steps'], ep['found_step'], ep['finder_id'], ep['delivery_step'], ep['success'], ep['terminated'],
                 ep['termination_reason']) == (outcome['steps'], outcome['found_step'], outcome['registered_finder'],
                 outcome['receipt_step'], outcome['S'], not outcome['censored'], outcome['reason']), 'collector labels')
        require(ep['rewards'] == [r['team'] for r in branch['intervals']] and
                math.isclose(branch['native_discounted'], outcome['observed_discounted_return'], abs_tol=1e-9) and
                math.isclose(ep['team_return'], math.fsum(ep['rewards']), abs_tol=1e-9), 'return reconstruction')
        require(len(ep['frames']) == ep['steps'] and all(f['positions'] == r['event']['agent_end'] and
                f['step'] == r['step'] for f, r in zip(ep['frames'], branch['intervals'])), 'physical frame record')
        rows[arm] = dict(labels=outcome, admissions=admissions(branch))
    return dict(arms=rows, request_digest=digest(req), prefix_digest=digest(prefix), branch_sha256=hashes,
                auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                counts=dict(assigned_branches=len(rows), native_terminal=sum(not r['labels']['censored'] for r in rows.values()),
                            physical_steps=sum(r['labels']['steps'] for r in rows.values())))


def compare_orders(left, right):
    a, b = audit_folder(left), audit_folder(right)
    reqs = [json.loads((p / 'request.json').read_bytes()) for p in (left, right)]
    require(reqs[0]['order'] == reqs[1]['order'][::-1], 'expected opposite traversal')
    require({k:v for k,v in reqs[0].items() if k != 'order'} ==
            {k:v for k,v in reqs[1].items() if k != 'order'}, 'order must be the only changed setting')
    require(a['prefix_digest'] == b['prefix_digest'] and a['arms'] == b['arms'], 'order changed decisions or labels')
    for arm in a['arms']:
        records = [json.loads((p / (arm+'.json')).read_bytes()) for p in (left, right)]
        require(all(records[0][k] == records[1][k] for k in records[0] if k != 'request_digest'),
                'traversal changed a physical branch')


def compare_prefix(short, long):
    a, b = audit_folder(short), audit_folder(long)
    requests = [json.loads((p/'request.json').read_bytes()) for p in (short, long)]
    require(requests[0]['cap'] < requests[1]['cap'] and
            {k:v for k,v in requests[0].items() if k != 'cap'} ==
            {k:v for k,v in requests[1].items() if k != 'cap'}, 'cap must be the only changed setting')
    require(a['prefix_digest'] == b['prefix_digest'], 'different departure before truncation')
    for arm in a['arms']:
        x, y = [json.loads((p/(arm+'.json')).read_bytes()) for p in (short, long)]
        n = len(x['intervals'])
        require(x['intervals'] == y['intervals'][:n] and all(
                x['episode'][k] == y['episode'][k][:n] for k in ('frames','rewards')) and
                x['episode']['decisions'] == [d for d in y['episode']['decisions'] if d['step'] < n],
                'different observed physical prefix')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--compare-order', type=Path)
    parser.add_argument('--longer', type=Path, help='check this prefix against a longer run')
    args = parser.parse_args()
    result = audit_folder(args.folder)
    if args.compare_order:
        compare_orders(args.folder, args.compare_order)
    if args.longer:
        compare_prefix(args.folder, args.longer)
    print(json.dumps(result, ensure_ascii=False, indent=2))
