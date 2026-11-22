"""Recheck the Windows receipts, public map snapshots and historical controls without an environment."""
import sys
from pathlib import Path
from types import SimpleNamespace as View

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from chapter3_bser.path_option.path_context import Route
from chapter3_bser.path_option.source import sha
from chapter3_bser.r123_final_leg import connection_verdict
from chapter3_bser.reach_avoid.scan_geometry import RaySubcells, ScanRefinedGeometry
from tools.r123_final_leg import STAGES, compare, identity, match_prefix, measures, reference
from tools.r123_task_sync_experiment import read, require, write


def map_verdict(folder, query):
    with np.load(folder / query['public_map'], allow_pickle=False) as m:
        shape, spacing, origin = tuple(m['grid_shape']), m['spacing'], m['origin']
        grid = View(shape=shape, spacing=spacing, origin=origin,
                    cell_centers=origin + (np.array(list(np.ndindex(shape))) + .5) * spacing)
        occupancy = View(occupied_mask=m['occupied'], free_mask=m['free'], unknown_mask=m['unknown'])
        rays = RaySubcells(spacing * shape)
        for key in ('free', 'blocked'):
            bits = np.unpackbits(m['ray_' + key + '_packbits'])[:np.prod(rays.shape)].reshape(rays.shape)
            setattr(rays, key, set(map(tuple, np.argwhere(bits))))
        hits, padding = m['scan_hits'], float(m['padding'])
    geometry = ScanRefinedGeometry(View(grid=grid, occupancy=occupancy), View(points=lambda: hits),
                                   dict(clearance=padding, hit_padding=0.), rays)
    data = query['carries'][0]['fresh_after']
    route = Route(data['points'], data['revision'], data['query_step'], data['reachable'], data['failure'],
                  index=data['index'])
    require(route.route_id == data['id'], 'wrong archived grid path')
    goal = query['requested_endpoint']
    return connection_verdict(View(route=route, task=goal, endpoint=goal, mode='TASK'), query['position'], geometry)


def verify():
    hist = read(HERE / 'historical_controls.json')
    source = identity()
    initial = read(HERE / 'search_prefix_source.json')
    runtime_file = 'chapter3_bser/r123_final_leg.py'
    require(sha(ROOT / runtime_file) == initial['files'][runtime_file], 'search-prefix runtime differs')
    require(hist['source'] == initial['sha256'], 'unexpected archive verification version')
    roots = [ROOT / '3090结果', ROOT / '3090结果/11.20']
    refs = {r['scene']: r for r in hist['controls']}
    require(len(refs) == 14, 'expected fourteen independent scenarios')
    prefix = {}
    for stage, cases in STAGES.items():
        for group, index in cases:
            _, scene, _, _, archived, metric, origin = reference(group, index, roots)
            key = scene['scenario_id']
            require(refs[key] == dict(stage=stage, scene=key, reference=origin,
                    measures=measures(archived, scene, metric.get('wall_seconds'))), 'archive measurement changed')
            if stage == 'focus':
                runs = [read(HERE / 'native' / f'{index:04d}_{arm}.json') for arm in ('grid', 'direct')]
                for raw in runs:
                    match_prefix(archived, raw, 8)
                prefix[key] = compare(*runs, measures(runs[1], dict(scene, max_steps=8), None))
                require(prefix[key] == 8, 'partial search prefix')
    base = roots[0] / 'r123_capture_0013_0017_20261122_171357_8PaKt4' / 'diagnostics'
    seal = base / 'completed.json'
    require(sha(seal) == '3df5ed279e2bec2530f235b70fdbe35b57e9b7427013de0c02a0870b86113009',
            'capture replay receipt changed')
    for name, expected in read(seal)['artifacts'].items():
        require(sha(base / name) == expected, 'archive changed: ' + name)
    maps = []
    for path in sorted(base.glob('*/q*_step*.json')):
        query = read(path)
        maps.append(dict(scene=path.parent.name, action=query['step'], verdict=map_verdict(path.parent, query),
                         query_sha256=sha(path), map_sha256=sha(path.parent / query['public_map'])))
    require(len(maps) == 7 and [row['verdict'] for row in maps] == ['outside_final_leg'] + ['direct'] * 6,
            'archived snapshot eligibility changed')
    near = read(HERE / 'native_near/completed.json')
    require(near['source'] == source and near['passed'], 'wrong near-field receipt')
    for record in near['records']:
        require(sha(HERE / 'native_near' / record['raw_file']) == record['raw_sha256'], 'changed fixture trace')
    protected = read(ROOT / 'verification/r123_expansion_v1/20261119_v1/before.json')
    ws = ROOT.parent
    counts = {}
    for name in ('CH3', 'CH4', 'CH5'):
        old = {k: v['sha256'] for k, v in protected.items() if k.startswith(name + '/')}
        now = {p.relative_to(ws).as_posix(): sha(p) for p in (ws / name).rglob('*') if p.is_file()}
        require(now == old, 'protected files changed: ' + name)
        counts[name] = len(now)
    return dict(source=source, runtime_sha256=sha(ROOT / runtime_file), initial_driver=initial['sha256'],
                archived_controls_checked=len(refs), original_capture_maps=maps, prefix_steps_per_arm=prefix,
                native_near_receipt_sha256=sha(HERE / 'native_near/completed.json'),
                native_completed_episodes=6, native_environment_initialization_attempts=7,
                rejected_fixture_resets_before_first_action=1, physical_steps=2 * sum(prefix.values()) + near['physical_steps'],
                training_updates=0, checkpoint_load=False, protected_sha256_sets_verified=counts)


if __name__ == '__main__':
    result = verify()
    path = HERE / 'windows_checks.json'
    if path.exists():
        require(read(path) == result, 'existing Windows evidence differs')
    else:
        write(path, result)
    print(result)
