"""Read a returned Linux pilot in place, including an incomplete transfer."""
import argparse
import json
from pathlib import Path
from statistics import median

import torch

from project_paths import ROOT
from tools.launch_assignment_audit import admissions, labels, require
from tools.launch_assignment_study import CONFIG, digest, file_hash, read
from tools.launch_development_pilot_audit import coverage, departure_actions, family_prefix, free_centers
from tools.launch_event_study import PILOT_HASH, STUDY, assess, identity, prediction, save_once


def review(folder):
    plan, provenance = identity()
    require(file_hash(folder/'plan.json') == PILOT_HASH, 'different returned pilot plan')
    sealed = read(STUDY/'predictions.json')
    require(all(sealed[k] == v for k,v in provenance.items()), 'different sealed model')
    runtime = read(folder/'runtime.json')
    require(runtime['runtime']['system'] == 'Linux' and runtime['plan_sha256'] == PILOT_HASH and
            runtime['source_lock'] == plan['source_lock'], 'different collection runtime')
    for name in ('tools/run_launch_development_pilot.py','tools/launch_development_pilot_audit.py'):
        require(runtime['orchestration_sha256'][name] == file_hash(ROOT/name), 'unrecognized runner')
    study = read(CONFIG)
    scene_file = (ROOT/study['native_inputs']['config']).with_name('scenes.json')
    scenes = {s['scenario_id']:s for s in read(scene_file)['scenarios']}
    totals = dict(branches=0, complete_blocks=0, physical_steps=0)
    families, missing = {}, []
    for f in plan['families']:
        name = f['scene']
        dep, = [d for d in runtime['departures'] if d['scene'] == name]
        ref = dep['snapshot']
        require(dep['family'] == f['family'] and dep['prefix_sha256'] == family_prefix(ref)
                and ref['initial']['scene_digest'] == digest(scenes[name]), 'different scenario or prefix')
        centers = free_centers(scenes[name])
        forecast = prediction(dep)
        prospective = name in sealed['predictions']
        matches = prospective and forecast == sealed['predictions'][name]['prediction']
        fixed_actions, blocks = {}, []
        for repeat in plan['repeats']:
            sub = folder / f"{name[-4:]}_r{repeat}"
            if not (sub/'request.json').exists():
                missing.append(sub.name+'/request.json')
                continue
            req = read(sub/'request.json')
            require((req['scene'],req['family'],req['repeat'],req['split'],req['fixture'],req['cap'],req['gamma'],
                     req['source_lock'],req['continuation'],req['terminal'],req['order'],req['assignment_axes'],
                     req['third_option'],req['actor_reference']) ==
                    (name,f['family'],repeat,'development',False,400,.95,plan['source_lock'],plan['continuation'],
                     plan['terminal'],[0,1,2,3],[0,1],0,study['candidates']['proposer_reference']), 'different request')
            audited, hashes, block, common = {}, {}, {}, None
            for arm in plan['arms']:
                path = sub/(arm+'.json')
                if not path.exists():
                    missing.append(sub.name+'/'+path.name)
                    continue
                hashes[arm] = file_hash(path)
                b = read(path)
                require(b['arm'] == arm and b['assignment'] == [int(arm[0]),int(arm[1]),0] and
                        b['request_digest'] == digest(req) and family_prefix(b) == dep['prefix_sha256'] and
                        b['actor_sha256'] == plan['actor_sha256'], 'wrong branch identity')
                prefix = digest([b[k] for k in ('initial','prefix','protocol','actor_sha256')])
                require(common in (None,prefix), 'unpaired branches')
                common = prefix
                for boat, (before,after) in enumerate(zip(b['prefix']['streams_before'],b['prefix']['streams_after'])):
                    seed = int(digest([study['random_streams']['master_seed'], f['family'],
                                       'continuation', boat, repeat])[:16],16)
                    import hashlib
                    state = torch.Generator().manual_seed(seed).get_state().numpy().tobytes()
                    require(before[2] == after[2] == hashlib.sha256(state).hexdigest(), 'wrong continuation stream')
                action = departure_actions(b)
                require(fixed_actions.setdefault(arm,action) == action, 'departure changed between repeats')
                y = labels(b['initial'], b['intervals'], b['protocol'], req['cap'])
                require(not y['censored'] and type(y['S']) is bool and b['episode']['success'] == y['S']
                        and b['episode']['steps'] == y['steps'], 'missing or inconsistent terminal')
                audited[arm] = dict(labels=y, admissions=admissions(b))
                geo = coverage(b, centers)
                block[arm] = dict(S=y['S'],found=y['ever_found'],received=y['ever_received'],reason=y['reason'],
                                  margin_m=geo['closest_margin_m'], initial_cells=geo['centers_until_first_continuation'],
                                  total_cells=geo['centers_ever_covered'], free_cells=geo['free_grid_centers'])
                totals['branches'] += 1
                totals['physical_steps'] += y['steps']
            ap = sub/'audit.json'
            if ap.exists():
                a = read(ap)
                require(a['auditor_sha256'] == file_hash(ROOT/'tools/launch_assignment_audit.py') and
                        a['request_digest'] == digest(req) and a['branch_sha256'] == hashes and
                        a['prefix_digest'] == common and a['arms'] == audited and len(audited) == 4,
                        'Linux audit differs from reconstructed branch labels or identities')
                totals['complete_blocks'] += 1
            else:
                missing.append(sub.name+'/audit.json')
            blocks.append(dict(repeat=repeat, audited_block=ap.exists(), arms=block, raw_sha256=hashes))
        rows = [r for block in blocks for r in block['arms'].values()]
        counts = {k:sum(r[k] for r in rows) for k in ('found','received','S')}
        contrasts = {}
        if matches and forecast['status'] == 'development_prediction':
            for b in ('0','1'):
                pairs = [r for r in blocks if r['audited_block']]
                if pairs:
                    contrasts[b] = assess([int(r['arms']['1'+b]['S'])-int(r['arms']['0'+b]['S'])
                                           for r in pairs], forecast['contrasts'][b]['predicted_sign'])
        margins = [r['margin_m'] for r in rows if r['margin_m'] is not None]
        families[name] = dict(counts=counts, n=len(rows), predicted_before_inspection=prospective,
                              runtime_prediction_matches=matches if prospective else None, contrasts=contrasts,
                              closest_margin_m=min(margins, default=None),
                              median_initial_cells=median(r['initial_cells'] for r in rows) if rows else None,
                              median_total_cells=median(r['total_cells'] for r in rows) if rows else None,
                              discordant_complete_blocks=[r['repeat'] for r in blocks if r['audited_block']
                                                         and len({v['S'] for v in r['arms'].values()}) > 1],
                              blocks=blocks)
    return dict(schema='auv.launch_events.return.v1', seal_sha256=file_hash(STUDY/'predictions.json'),
                runtime_sha256=file_hash(folder/'runtime.json'), audit_sha256=file_hash(Path(__file__)),
                totals=totals, missing=missing, complete=(totals['complete_blocks']==12 and not missing),
                families=families, linux_summary_present=(folder/'summary.json').is_file())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = review(args.folder)
    save_once(args.output, result)
    print(json.dumps(dict(totals=result['totals'],missing=result['missing'],
                          families={k:v['counts'] for k,v in result['families'].items()})))
