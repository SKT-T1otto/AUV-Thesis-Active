"""Paired, rule-only evaluation of the capture final leg on frozen Chapter 3 scenes."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from chapter3_bser.path_option.source import digest, sha
from chapter3_bser.r123_final_leg import RuleRuntime
from core.env.target_motion import TargetState, simulate_target_trajectory, swept_relative_min_distance
from project_paths import ROOT
from tools.r123_expansion import audit_episode, identity as parent_identity
from tools.r123_task_sync_experiment import fresh, read, require, rows, write


HISTORY = {
    'expansion': ('r123_expansion_20261120_123244_3078331', 'evaluation',
                  '00a9d3cb364b0066a272c55a057188fcc49172108c52b08c4eb33f2d36d92ff4'),
    'original_first': ('r123_task_sync_20261118_223936_3078331', 'sync',
                       'bb2713360ddce0c470c8df8006ac412559f778bf062f07e86a90dc0d15c13928'),
    'original_last': ('r123_task_sync_regression_20261119_161420_3078331', 'sync',
                      '96f8c0f86ba5b54bd197115d1f75142201e4f92b176813a1a45cabc59ee10359'),
}
STAGES = {
    'focus': [('expansion', 13), ('expansion', 17)],
    'regression': [('expansion', i) for i in (1, 6, 7, 10, 11, 15, 21, 22)]
                  + [('original_first', 1)] + [('original_last', i) for i in (3, 4, 5)],
}
MANIFEST = ROOT / 'provenance/r123_final_leg_manifest_v1.json'


def identity():
    m = read(MANIFEST)
    require(digest({k: v for k, v in m.items() if k != 'sha256'}) == m['sha256'], 'invalid final-leg manifest')
    require(parent_identity()['sha256'] == m['parent_expansion_sha256'], 'parent source changed')
    for name, expected in m['files'].items():
        require(sha(ROOT / name) == expected, 'source changed: ' + name)
    return m['sha256']


def reference(group, index, roots):
    """Read only the plan and original task-sync episode authenticated by its Linux receipt."""
    name, subdir, expected = HISTORY[group]
    run = next((r / name for r in roots if (r / name / subdir / 'completed.json').is_file()), None)
    require(run is not None, f'missing historical run {name} under {roots}')
    result = run / subdir
    receipt = result / 'completed.json'
    require(sha(receipt) == expected, f'wrong completion receipt: {receipt}')
    done = read(receipt)
    require(not done['training'] and not done['checkpoint_load'], 'historical run is not rule-only')
    plan_dir = run / 'plan'
    require(sha(plan_dir / 'plan.json') == done['plan_sha256'], 'historical plan changed')
    plan = read(plan_dir / 'plan.json')
    for item in ('config.json', 'options.json', 'scenes.json'):
        require(sha(plan_dir / item) == plan['inputs'][item], f'historical input changed: {item}')
    config, options, scenes = [read(plan_dir / item) for item in ('config.json', 'options.json', 'scenes.json')]
    if group == 'expansion':
        scene = scenes['scenarios'][index]
        prefix = f'{index:04d}_task_sync/'
        episode_file, metric_file = prefix + 'episode.json', prefix + 'metric.json'
        seed = plan['evaluation_seed'] + index
        scenario_id = f'r123_expand_20261119_{index:04d}'
    else:
        scene = next(s for s in scenes if s['scenario_id'] == f'po_eval_992001_{index:04d}')
        episode_file, metric_file = 'episodes.jsonl', 'metrics.jsonl'
        seed = plan['seed'] + index
        scenario_id = f'po_eval_992001_{index:04d}'
    require(scene['scenario_id'] == scenario_id and config['max_steps'] == scene['max_steps'] == 400
            and scene['target_capture_radius'] == .8 and options['gamma'] == .95, 'wrong scenario or contract')
    artifacts = done['artifacts']
    for key in (episode_file, metric_file):
        require(sha(result / key) == artifacts[key], f'historical artifact changed: {key}')
    if group == 'expansion':
        raw, metric = read(result / episode_file), read(result / metric_file)
    else:
        def selected(key):
            matching = [r for r in rows(result / key) if r['scenario_id'] == scenario_id]
            require(len(matching) == 1, 'missing or duplicated archived case')
            return matching[0]
        raw, metric = selected(episode_file), selected(metric_file)
    require(raw['scenario_id'] == metric['scenario_id'] == scenario_id and raw['episode_id'] == index
            and raw['task_sync']['enabled'] and not raw['task_network_trained'], 'wrong historical arm')
    provenance = dict(run=name, completion_sha256=expected, plan_sha256=done['plan_sha256'],
                      episode_file=episode_file, episode_file_sha256=artifacts[episode_file], seed=seed)
    return config, scene, options, seed, raw, metric, provenance


def simulate(config, scene, options, index, seed, direct):
    import torch
    from chapter3_bser.learning.reference import ZeroResidualPolicy

    limit = 400 if platform.system() == 'Linux' else 32
    require(0 < config['max_steps'] == scene['max_steps'] <= limit, 'invalid native episode budget')
    torch.set_num_threads(1)
    start = time.monotonic()
    r = RuleRuntime(config, scene, seed=seed, episode_id=index, options=options, arm='rule',
                    tracking_arm='both', task_sync=True, direct_final_leg=direct)
    try:
        policy = ZeroResidualPolicy()
        while not r.terminal:
            r.advance(policy, explore=False)
            require(r.step <= config['max_steps'], 'step cap exceeded')
            if r.step % 25 == 0:
                print(f'{scene["scenario_id"]} direct={direct} step={r.step}/{config["max_steps"]}', flush=True)
        raw = r.summary()
        require(raw['final_leg']['enabled'] is direct and raw['task_sync']['enabled'] and not raw['task_network_trained']
                and raw['task_sync']['receipt_violations'] == 0, 'rule/sync contract violated')
        return raw, time.monotonic() - start
    finally:
        r.close()


def measures(raw, scene, wall_seconds):
    audit_episode(raw, scene)  # Independently checks discovery, handoff, contacts and consecutive full intervals.
    target = simulate_target_trajectory(
        TargetState(np.array(scene['target_initial_position'], np.float32), scene['target_initial_velocity'],
                    0, scene['target_motion_mode']), raw['actual_length'], .2, (20., 20., 8.),
        scene['obstacles'], clearance=scene['target_obstacle_clearance'])
    trace = raw['path_option']['trace']
    capture = [r for r in trace if r['agent'] == 3 and r['phase'] == 2]
    distances = {}
    for r in capture:
        step = r['step']
        distances[step] = float(swept_relative_min_distance(
            r['position'], r['next_position'], target[step].position, target[step+1].position)[0])
    leg = raw.get('final_leg', {})
    direct_steps = []
    if leg.get('enabled'):
        decisions = leg['steps']
        require(len(decisions) == len(capture), 'missing final-leg decisions')
        for d, r in zip(decisions, capture):
            require((d['step'], d['route_id'], d['selected_index'], d['map_revision'])
                    == (r['step'], r['route_id'], r['route_index'], r['map_revision']), 'cursor/trace mismatch')
            if d['verdict'] == 'direct':
                require(d['grid_index'] < d['selected_index']
                        and np.allclose(r['event_target'], r['endpoint'], atol=1e-4, rtol=0), 'wrong final point')
                direct_steps.append(d['step'])
    eligible = [r for r in trace if r['phase'] != 3]
    return dict(steps=raw['actual_length'], found_step=raw['found_step'],
                handoff_step=raw['executor_target_received_step'], success=bool(raw['success']),
                success_step=raw['success_step'], collision=bool(raw['collision_episode']),
                closest_capture_m=min(distances.values(), default=None),
                first_contact_step=next((t + 1 for t, d in distances.items() if d <= scene['target_capture_radius']), None),
                contact_steps=raw['capture_contact_step_count'], max_consecutive_hold=raw['capture_hold_counter_max'],
                capture_brakes=sum(r['decision']['action'] == 8 for r in capture),
                capture_blocked=sum(r['navigation_blocked'] for r in capture),
                emergency_actions=sum(r['emergency'] for r in eligible),
                first_direct_action=min(direct_steps, default=None), direct_actions=len(direct_steps),
                route_queries=raw['task_sync']['route_queries'], wall_seconds=wall_seconds)


def match_prefix(reference_raw, actual, count):
    """All four agents, through the last action preceding the first shortcut."""
    old, new = [r['path_option']['trace'] for r in (reference_raw, actual)]
    require(count >= 0 and min(len(old), len(new)) >= count * 4, 'short comparison prefix')
    discrete = ('step', 'agent', 'phase', 'mode', 'route_id', 'route_index', 'reason', 'emergency')
    physical = ('position', 'next_position', 'velocity', 'executed', 'event_target', 'endpoint')
    for a, b in zip(old[:4*count], new[:4*count]):
        require(all(a[k] == b[k] for k in discrete) and a['decision']['action'] == b['decision']['action'],
                f'prefix decisions differ at action {a["step"]} agent {a["agent"]}')
        require(np.allclose([a[k] for k in physical], [b[k] for k in physical], atol=1e-6, rtol=0),
                f'prefix physics differ at action {a["step"]} agent {a["agent"]}')


def compare(control, candidate, measured):
    require(candidate['final_leg']['enabled'] and not control.get('final_leg', {}).get('enabled', False),
            'expected synchronized grid control and direct candidate')
    first = measured['first_direct_action']
    if first is None:
        require(control['actual_length'] == candidate['actual_length'], 'length changed with no direct action')
    require(control['found_step'] == candidate['found_step']
            and control['executor_target_received_step'] == candidate['executor_target_received_step'],
            'discovery or handoff changed')
    count = candidate['actual_length'] if first is None else first
    match_prefix(control, candidate, count)
    return count


def completed(stage_dir, stage, source):
    d = read(stage_dir / 'completed.json')
    require(d['source'] == source and d['stage'] == stage and d['cases'] == len(STAGES[stage]),
            'wrong stage completion')
    for name, expected in d['artifacts'].items():
        require(sha(stage_dir / name) == expected, 'changed stage output: ' + name)
    return d


def evaluate(stage, output, roots):
    require(platform.system() == 'Linux', 'full episodes run manually on Linux')
    source = identity()
    if stage == 'regression':
        completed(output / 'focus', 'focus', source)
    cases = STAGES[stage]
    # Check every historical input before spending steps on the first pair.
    for group, index in cases:
        reference(group, index, roots)
    dest = fresh(output / stage)
    physical_steps = 0
    pairs = []
    checksums = {}
    def save(path, data):
        write(path, data)
        checksums[path.relative_to(dest).as_posix()] = sha(path)

    budget = len(cases) * (2 if stage == 'focus' else 1) * 400
    save(dest / 'request.json', dict(source=source, stage=stage, cases=cases, max_steps_per_episode=400,
                                    maximum_new_steps=budget, training=False, checkpoint_load=False))
    for number, (group, index) in enumerate(cases, 1):
        cfg, scene, opt, seed, old, old_metric, origin = reference(group, index, roots)
        folder = fresh(dest / scene['scenario_id'])
        baseline = measures(old, scene, old_metric.get('wall_seconds'))
        expected_success = stage == 'regression' and (group, index) != ('original_last', 4)
        require(baseline['success'] == expected_success, 'wrong historical outcome')
        record = dict(scene=scene['scenario_id'], reference=origin, archived_grid=baseline)
        if stage == 'focus':
            grid, elapsed = simulate(cfg, scene, opt, index, seed, False)
            physical_steps += grid['actual_length']
            save(folder / 'grid_episode.json', grid)
            record['fresh_grid'] = measures(grid, scene, elapsed)
            require(grid['actual_length'] == old['actual_length'], 'baseline replay length differs')
            match_prefix(old, grid, grid['actual_length'])
            old = grid
        candidate, elapsed = simulate(cfg, scene, opt, index, seed, True)
        physical_steps += candidate['actual_length']
        save(folder / 'direct_episode.json', candidate)
        record['direct'] = measures(candidate, scene, elapsed)
        record['matched_prefix_steps'] = compare(old, candidate, record['direct'])
        save(folder / 'pair.json', record)
        pairs.append(record)
        print(f'[{stage} {number}/{len(cases)}] {scene["scenario_id"]} '
              f'grid_success={baseline["success"]} direct={json.dumps(record["direct"])}', flush=True)
    require(identity() == source and physical_steps <= budget, 'source or budget changed')
    save(dest / 'report.json', dict(pairs=pairs, new_physical_steps=physical_steps,
         rescued=[p['scene'] for p in pairs if p['direct']['success'] and not p['archived_grid']['success']],
         lost=[p['scene'] for p in pairs if p['archived_grid']['success'] and not p['direct']['success']]))
    write(dest / 'completed.json', dict(stage=stage, cases=len(pairs), source=source,
          episodes=len(cases) * (2 if stage == 'focus' else 1), new_physical_steps=physical_steps,
          training_updates=0, checkpoint_load=False, artifacts=checksums))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--output', type=Path, required=True, help='fresh run root shared by both stages')
    parser.add_argument('--history-root', type=Path, action='append', help='parent of original Linux run folders')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        evaluate(args.stage, args.output, args.history_root or [ROOT / 'work/collision_terminal'])
    else:
        print(json.dumps(dict(stage=args.stage, cases=STAGES[args.stage], executed=False, source=identity())))


if __name__ == '__main__':
    main()
