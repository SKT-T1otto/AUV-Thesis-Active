"""Frozen 24-scene paired rule screen; full evaluation is manual Linux only."""
import argparse
import copy
import json
import math
from pathlib import Path
import platform
import time
import traceback
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

from project_paths import ROOT
from chapter3_bser.path_option.source import sha, digest
from chapter3_bser.r123_task_sync.source import identity as control_identity, parent_identity
from tools.r123_task_sync_regression import identity as observer_identity, validation
from tools.r123_task_sync_experiment import read, rows, write, journal, fresh, require, task_metrics
from chapter3_bser.learning.config import scenario_fingerprint
from chapter3_bser.learning.scenarios import validate_manifest

DATA = ROOT / 'configs/r123_expansion_v1'
MANIFEST = ROOT / 'provenance/r123_expansion_manifest_v1.json'
ARMS = ('tracking_both', 'task_sync')
COUNT = 24
STEP_CAP = 400
EVALUATION_SEED = 20261119
MAXIMUM_STEPS = COUNT * len(ARMS) * STEP_CAP


def identity():
    m = read(MANIFEST)
    require(digest({k: v for k, v in m.items() if k != 'sha256'}) == m['sha256'], 'expansion manifest invalid')
    require(m['control_source'] == control_identity() and m['baseline_source'] == parent_identity(), 'control source changed')
    require(m['observer_source'] == observer_identity(), 'read-only observer changed')
    for name, h in m['files'].items():
        require(sha(ROOT / name) == h, 'expansion file changed: ' + name)
    return dict(schema='auv.r123.expansion.tool.v1', sha256=m['sha256'],
                control_source=m['control_source'], baseline_source=m['baseline_source'], observer_source=m['observer_source'])


def contract(config, options, manifest, exclusions):
    scenes = validate_manifest(manifest, config, evaluation=True)
    require(len(scenes) == COUNT and config['max_steps'] == STEP_CAP and options['gamma'] == .95, '24 scenes x400 required')
    require([s['scenario_id'] for s in scenes] == [f'r123_expand_20261119_{n:04d}' for n in range(COUNT)], 'wrong expansion order')
    fingerprints = [scenario_fingerprint(s) for s in scenes]
    require(not set(fingerprints) & set(exclusions['fingerprints']), 'physical overlap with prior manifests')
    require(len(set(fingerprints)) == COUNT, 'duplicate physical scenes')
    require(all(s['obstacle_knowledge_mode'] == 'online_unknown' and s['target_motion_mode'] == 'constant_velocity_reflect_v1'
                and s['target_capture_radius'] == .8 and s['target_trajectory_reachable'] for s in scenes), 'mission contract changed')
    return scenes


def jobs():
    result = []
    for n in range(COUNT):
        order = ARMS if n % 2 == 0 else tuple(reversed(ARMS))
        result.extend(dict(scene=n, arm=arm, seed=EVALUATION_SEED + n, folder=f'{n:04d}_{arm}') for arm in order)
    return result


def prepare(output):
    source = identity()
    files = {n: read(DATA / n) for n in ('config.json', 'options.json', 'scenes.json', 'exclusions.json')}
    contract(files['config.json'], files['options.json'], files['scenes.json'], files['exclusions.json'])
    out = fresh(output)
    for name, value in files.items():
        write(out / name, value)
    write(out / 'plan.json', dict(tool=source, count=COUNT, arms=list(ARMS), max_steps=STEP_CAP,
        maximum_new_physical_steps=MAXIMUM_STEPS, evaluation_seed=EVALUATION_SEED,
        jobs=jobs(), training=False, checkpoint_load=False, baseline_reused=False,
        purpose='expanded development screen; old six retained separately; not final test',
        inputs={name: sha(out / name) for name in files}))
    print(json.dumps(dict(prepared=True, scenes=COUNT, jobs=48, maximum_physical_steps=MAXIMUM_STEPS, new_physical_steps=0)))


def load_plan(root):
    root = Path(root)
    p = read(root / 'plan.json')
    require(p['tool'] == identity() and p['count'] == COUNT and p['jobs'] == jobs(), 'wrong source/count/jobs')
    require(p['maximum_new_physical_steps'] == MAXIMUM_STEPS and p['max_steps'] == STEP_CAP and
            p['evaluation_seed'] == EVALUATION_SEED and p['arms'] == list(ARMS), 'budget/arms changed')
    require(not p['training'] and not p['checkpoint_load'] and not p['baseline_reused'], 'rule-only new paired baseline required')
    for name, h in p['inputs'].items():
        require(sha(root / name) == h and sha(DATA / name) == h, 'plan or frozen input changed: ' + name)
    c, o, manifest, x = (read(root / name) for name in ('config.json', 'options.json', 'scenes.json', 'exclusions.json'))
    return p, c, o, contract(c, o, manifest, x)


def run_one(config, scene, options, index, arm, *, ledger, smoke=False, plain=False):
    import torch
    from chapter3_bser.learning.reference import ZeroResidualPolicy
    from chapter3_bser.r123_tracking.runtime import RuleRuntime as Tracking
    from chapter3_bser.r123_task_sync.runtime import RuleRuntime as Sync
    from chapter3_bser.r123_task_sync.diagnostics import ObservedRuntime
    from chapter3_bser.r123_tracking.workflow import metric
    require(arm in ARMS, 'unknown arm')
    require(smoke or platform.system() == 'Linux', 'full episodes are manual Linux only')
    require(0 < config['max_steps'] <= (32 if smoke else STEP_CAP) and scene['max_steps'] == config['max_steps'], 'episode limit mismatch')
    require(smoke or config['max_steps'] == STEP_CAP, 'full cases require400-step cap')
    torch.set_num_threads(1)
    ledger['environment_initializations'] += 1
    runtime = None
    start = time.monotonic()
    try:
        cls = (Tracking if arm == 'tracking_both' else Sync) if plain else ObservedRuntime
        kwargs = dict(seed=EVALUATION_SEED + index, episode_id=index, options=options, arm='rule', tracking_arm='both')
        if cls is not Tracking:
            kwargs['task_sync'] = arm == 'task_sync'
        runtime = cls(config, scene, **kwargs)
        policy = ZeroResidualPolicy()
        while not runtime.terminal:
            runtime.advance(policy, explore=False)
            require(runtime.step <= config['max_steps'], 'step cap exceeded')
            if runtime.step % 25 == 0:
                print(f'[expansion] scene={index + 1}/24 arm={arm} step={runtime.step}/{config["max_steps"]} training=0', flush=True)
        raw = runtime.summary()
        m = metric(raw, config['max_steps'])
        m.update(task_metrics(raw))
        sync = raw.get('task_sync', {})
        diagnostics = raw.get('failure_diagnostics', {})
        m.update(wall_seconds=time.monotonic() - start, receipt_violations=sync.get('receipt_violations', 0),
                 route_queries=sync.get('route_queries'), deployed_emergency_steps=sum(
                     r['emergency'] and r['phase'] != 3 for r in raw['path_option']['trace']),
                 failure_diagnostic_records=len(diagnostics.get('records', [])))
        require(not raw['task_network_trained'] and m['active_recovery_enters'] == 0 and m['receipt_violations'] == 0, 'unexpected training/recovery/receipt error')
        if not plain:
            require(raw['task_sync']['enabled'] == (arm == 'task_sync') and not diagnostics['changes_control'], 'arm or observer mismatch')
        return raw, m
    finally:
        if runtime is not None:
            ledger['physical_steps'] += runtime.step
            runtime.close()


def check(root, output, *, native=False):
    p, c, o, scenes = load_plan(root)
    out = fresh(output)
    result = dict(tool=identity(), plan_sha256=sha(Path(root) / 'plan.json'), platform=platform.system(),
        environment_initializations=0, physical_steps=0, training_updates=0, checkpoint_load=False,
        passed=False, native_requested=native)
    try:
        import checks.test_r123_expansion as tests
        import checks.test_r123_regression_diagnostics as observers
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in (tests, observers))
        with (out / 'synthetic.log').open('x', encoding='utf-8') as stream:
            with patch('core.env.mission_env.MissionCoreEnv.__init__', side_effect=AssertionError('no environment in synthetic checks')), \
                 redirect_stdout(stream), redirect_stderr(stream):
                r = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        result['synthetic'] = dict(tests=r.testsRun, failures=len(r.failures), errors=len(r.errors))
        require(r.wasSuccessful() and not r.skipped, 'synthetic checks failed')
        if native:
            c = copy.deepcopy(c)
            c['max_steps'] = 32
            s = copy.deepcopy(scenes[0])
            s['max_steps'] = 32
            traces = []
            for n, (arm, plain) in enumerate(((ARMS[0], True), (ARMS[0], False), (ARMS[1], True), (ARMS[1], False))):
                write(out / f'initialization_{n}.json', dict(arm=arm, plain=plain, steps=32, training=False))
                raw, m = run_one(c, s, o, 0, arm, ledger=result, smoke=True, plain=plain)
                write(out / f'episode_{n}.json', raw)
                write(out / f'metric_{n}.json', m)
                traces.append(raw['path_option']['trace'])
            require(traces[0] == traces[1] and traces[2] == traces[3], 'observer changed physical/decision trace')
            result['observer_trace_equivalence'] = True
        require(identity() == result['tool'], 'source changed during checks')
        result['passed'] = True
    except Exception:
        result['traceback'] = traceback.format_exc()
    result['artifacts'] = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    write(out / 'checks.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('tool', 'artifacts', 'traceback')}))
    require(result['passed'], 'checks failed; see receipt')


def evaluate(root, checks, validated, output):
    require(platform.system() == 'Linux', 'full expansion is manual Linux only')
    p, c, o, scenes = load_plan(root)
    validation(validated)  # Same immutable control source already validated on Linux.
    v = read(Path(checks) / 'checks.json')
    require(v['tool'] == identity() and v['passed'] and v['physical_steps'] == 0 and
            v['plan_sha256'] == sha(Path(root) / 'plan.json'), 'fresh zero-step tool checks required')
    for name, h in v['artifacts'].items():
        require(sha(Path(checks) / name) == h, 'check artifact changed')
    out = fresh(output)
    source = identity()
    ledger = dict(environment_initializations=0, physical_steps=0)
    write(out / 'request.json', dict(tool=source, plan_sha256=sha(Path(root) / 'plan.json'),
        inherited_linux_validation_sha256=sha(Path(validated) / 'validation/validation.json'),
        maximum_new_physical_steps=MAXIMUM_STEPS, training=False, checkpoint_load=False))
    try:
        for j, job in enumerate(p['jobs']):
            dest = fresh(out / job['folder'])
            write(dest / 'request.json', dict(tool=source, **job, max_steps=STEP_CAP))
            raw, m = run_one(c, scenes[job['scene']], o, job['scene'], job['arm'], ledger=ledger)
            write(dest / 'episode.json', raw)
            write(dest / 'metric.json', m)
            require(ledger['physical_steps'] <= MAXIMUM_STEPS and identity() == source, 'budget/source changed')
            write(dest / 'completed.json', dict(tool=source, **job, physical_steps=m['steps'],
                artifacts={f.name: sha(f) for f in dest.iterdir() if f.is_file()}))
            journal(out / 'metrics.jsonl', dict(**job, metric=m))
            write(out / 'progress.json', dict(completed_jobs=j + 1, expected_jobs=48, **ledger))
            print(f'[expansion] completed={j + 1}/48 scene={job["scene"]:04d} arm={job["arm"]} success={m["success"]} found={m["found"]} collision={m["collision"]}', flush=True)
        write(out / 'completed.json', dict(tool=source, jobs=48, scenarios=24, **ledger,
            plan_sha256=sha(Path(root) / 'plan.json'), training=False, checkpoint_load=False,
            artifacts={f.relative_to(out).as_posix(): sha(f) for f in out.rglob('*') if f.is_file()}))
    except Exception:
        write(out / 'failed.json', dict(completed=False, **ledger, traceback=traceback.format_exc(),
            notice='No automatic retry. Preserve incomplete case and account for consumed steps before another run.'))
        raise


def wilson(successes, count):
    require(type(successes) is int and type(count) is int and 0 <= successes <= count and count > 0, 'invalid binomial count')
    z = 1.959963984540054
    p = successes / count
    den = 1 + z * z / count
    center = (p + z * z / (2 * count)) / den
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / den
    return [max(0., center - half), min(1., center + half)]


def audit_episode(raw, scene):
    """Independently verify detection and full-interval hold, without an env."""
    import numpy as np
    from core.env.target_motion import TargetState, simulate_target_trajectory, swept_relative_min_distance
    maximum = scene['max_steps']
    steps = raw['actual_length']
    trace = raw['path_option']['trace']
    require(not raw['task_network_trained'] and 0 < steps <= maximum and len(trace) == steps * 4, 'raw count/training mismatch')
    require({(r['step'], r['agent']) for r in trace} == {(t, a) for t in range(steps) for a in range(4)}, 'raw steps/agents missing')
    target = simulate_target_trajectory(TargetState(np.asarray(scene['target_initial_position'], np.float32),
        scene['target_initial_velocity'], 0, scene['target_motion_mode']), maximum, .2,
        (20., 20., 8.), scene.get('obstacles', []), clearance=.2)
    detected = []
    contacts = hold = max_hold = 0
    curve = np.zeros((maximum + 1, 4), np.int64)
    for r in trace:
        t, a = r['step'], r['agent']
        if a < 3 and r['phase'] == 0:
            distance, _ = swept_relative_min_distance(r['position'], r['next_position'], target[t].position, target[t + 1].position)
            if distance <= (2.1, 2.45, 2.85)[a]:
                detected.append(t + 1)
            curve[t + 1] += [1, int(np.linalg.norm(r['velocity']) < .05 and np.linalg.norm(np.asarray(r['position']) - r['endpoint']) > .8),
                             int(r['decision']['action'] == 8), int(r['reason'] == 'WAIT_TASK')]
        if a == 3 and r['phase'] == 2:
            distance, _ = swept_relative_min_distance(r['position'], r['next_position'], target[t].position, target[t + 1].position)
            contacts += int(distance <= .8)
            full = max(np.linalg.norm(np.asarray(r['position']) - target[t].position),
                       np.linalg.norm(np.asarray(r['next_position']) - target[t + 1].position)) <= .8
            hold = hold + 1 if full else 0
            max_hold = max(max_hold, hold)
    require(min(detected, default=None) == raw['found_step'], 'found step does not match actual sweep')
    require(contacts == raw['capture_contact_step_count'] and max_hold == raw['capture_hold_counter_max'], 'contact/hold mismatch')
    require(bool(max_hold >= 5) == bool(raw['success']), 'success does not match original hold rule')
    if raw['found_step'] is not None:
        require(raw['executor_target_received_step'] == raw['found_step'] + 1, 'handoff clock mismatch')
    return np.cumsum(curve, axis=0)


def report(root, results, output):
    p, c, o, scenes = load_plan(root)
    base = Path(results)
    done = read(base / 'completed.json')
    require(done['tool'] == identity() and done['jobs'] == 48 and done['scenarios'] == 24 and
            done['plan_sha256'] == sha(Path(root) / 'plan.json') and done['physical_steps'] <= MAXIMUM_STEPS, 'incomplete or wrong run')
    for name, h in done['artifacts'].items():
        require(sha(base / name) == h, 'result changed: ' + name)
    by_case, curves = {}, {}
    for job in p['jobs']:
        folder = base / job['folder']
        completed = read(folder / 'completed.json')
        require(completed['tool'] == identity() and all(completed[k] == job[k] for k in ('arm', 'scene', 'seed', 'folder')), 'job pairing changed')
        raw, m = read(folder / 'episode.json'), read(folder / 'metric.json')
        require(raw['scenario_id'] == scenes[job['scene']]['scenario_id'] and not raw['task_network_trained'], 'wrong raw scene')
        require(raw['task_sync']['enabled'] == (job['arm'] == 'task_sync'), 'arm label changed')
        from chapter3_bser.r123_tracking.workflow import metric
        import numpy as np
        recomputed = dict(metric(raw, STEP_CAP), **task_metrics(raw))
        for key, value in recomputed.items():
            equal = bool(np.isclose(value, m[key], rtol=1e-10, atol=1e-10)) if isinstance(value, (int, float)) else value == m[key]
            require(equal, 'metric mismatch: ' + key)
        curves[job['scene'], job['arm']] = audit_episode(raw, scenes[job['scene']])
        by_case.setdefault(job['scene'], {})[job['arm']] = m
    require(set(by_case) == set(range(COUNT)) and all(set(v) == set(ARMS) for v in by_case.values()), '24 complete pairs required')
    totals = {}
    for arm in ARMS:
        values = [by_case[i][arm] for i in range(COUNT)]
        successes = sum(int(v['success']) for v in values)
        totals[arm] = dict(episodes=COUNT, success=successes, found=sum(int(v['found']) for v in values),
            collision=sum(int(v['collision']) for v in values), success_interval95=wilson(successes, COUNT),
            physical_steps=sum(v['steps'] for v in values), low_speed=sum(v['search_slow_distant_steps'] for v in values),
            search_agent_steps=sum(v['search_agent_steps'] for v in values), emergencies=sum(v['deployed_emergency_steps'] for v in values))
    wins = sum(by_case[i]['task_sync']['success'] > by_case[i]['tracking_both']['success'] for i in by_case)
    losses = sum(by_case[i]['task_sync']['success'] < by_case[i]['tracking_both']['success'] for i in by_case)
    common_windows = []
    for i in range(COUNT):
        horizon = min(by_case[i][arm]['found_step'] or STEP_CAP for arm in ARMS)
        common_windows.append(dict(scene=i, horizon=horizon, arms={arm: dict(zip(
            ('search_agent_steps', 'low_speed', 'brake', 'wait'), map(int, curves[i, arm][horizon]))) for arm in ARMS}))
    out = fresh(output)
    write(out / 'report.json', dict(tool=identity(), complete_pairs=24, arms=totals,
        paired_success=dict(wins=wins, losses=losses, ties=24 - wins - losses),
        common_search_windows=common_windows,
        cases=[dict(scene=i, **by_case[i]) for i in range(COUNT)], performance_accepted=False,
        notice='New24 development screen only; old6 reported separately. Wilson intervals are descriptive; no formal held-out or cross-seed claim. Observer overhead and changing terminal lengths prevent wall-time speed claims.'))
    print(json.dumps(dict(report=str(out / 'report.json'), arms=totals, paired_wins=wins, paired_losses=losses)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'check', 'evaluate', 'report'))
    parser.add_argument('--root', type=Path)
    parser.add_argument('--checks', type=Path)
    parser.add_argument('--validated-run', type=Path)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-smoke', action='store_true')
    parser.add_argument('--execute', action='store_true')
    a = parser.parse_args()
    if not a.execute:
        print(json.dumps(dict(executed=False, command=a.command, maximum_evaluation_steps=MAXIMUM_STEPS)))
        return
    if a.command == 'prepare':
        prepare(a.output)
    else:
        require(a.root is not None, '--root required')
        if a.command == 'check':
            check(a.root, a.output, native=a.native_smoke)
        elif a.command == 'evaluate':
            require(a.checks is not None and a.validated_run is not None, '--checks and --validated-run required')
            evaluate(a.root, a.checks, a.validated_run, a.output)
        else:
            require(a.results is not None, '--results required')
            report(a.root, a.results, a.output)


if __name__ == '__main__':
    main()
