"""Read-only uploaded-trajectory audit. No environment or learner is created."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chapter3_bser.r123_governed.source import identity, digest
from chapter3_bser.r123_governed.workflow import metric
from core.env.target_motion import TargetState, simulate_target_trajectory, swept_relative_min_distance

OUT = Path(__file__).resolve().parent
INPUT = ROOT / '3090结果/r123_governed_20261116_215327_3078331'
OLD = ROOT / 'work/collision_terminal/governed_previous_input/no_recovery'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def snapshot_legacy():
    result = {}
    for folder in ('CH3', 'CH4', 'CH5'):
        root = ROOT.parent / folder
        result[folder] = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*'))
                          if p.is_file() and not any(x in p.parts for x in ('.git', '__pycache__', '.pytest_cache'))}
    return result


def same(a, b):
    if isinstance(a, (int, float, bool)) and isinstance(b, (int, float, bool)):
        return bool(np.isclose(a, b, rtol=1e-11, atol=1e-11))
    return a == b


def dist(a, b):
    return float(np.linalg.norm(np.asarray(a) - b))


def search_rows(raw):
    return [r for r in raw['path_option']['trace'] if r['agent'] < 3 and r['phase'] == 0]


def row_brief(r):
    d = r['action_diagnostics']
    return {**{k: r[k] for k in ('step', 'agent', 'phase', 'position', 'velocity', 'event_target',
                                'endpoint', 'route_id', 'route_revision', 'route_index', 'reason')},
            'speed': dist(r['velocity'], np.zeros(3)), 'tracking_distance': r['distance_to_tracking'],
            'speed_cap_xy': d['speed_cap_xy'], 'speed_cap_z': d['speed_cap_z'],
            'action': r['decision']['action'], 'navigation_candidates': d['navigation_candidates'],
            'action_checks': Counter(x['reason'] for x in d['candidate_checks'])}


def proximity(raw, trajectory):
    values = []
    for r in search_rows(raw):
        step = r['step']
        d, _ = swept_relative_min_distance(r['position'], r['next_position'],
                                          trajectory[step].position, trajectory[step + 1].position)
        values.append(dict(step=step + 1, agent=r['agent'], distance=d,
                           margin=d - (2.1, 2.45, 2.85)[r['agent']]))
    return values


before = dict(source=identity(), legacy=snapshot_legacy(),
              inputs={p.relative_to(ROOT).as_posix(): sha(p) for base in (INPUT, OLD)
                      for p in sorted(base.rglob('*')) if p.is_file()})
write('before.json', before)
completed = read(INPUT / 'governed/completed.json')
validation = read(INPUT / 'validation/validation.json')
assert completed['source'] == validation['source'] == before['source']
assert validation['platform'] == 'Linux' and validation['passed'] and validation['native_smoke_passed']
for name, expected in validation['artifacts'].items():
    assert sha(INPUT / 'validation' / name) == expected, name
for folder in (INPUT / 'governed', OLD):
    receipt = read(folder / 'completed.json')
    for name in ('summary', 'episodes', 'metrics'):
        assert sha(folder / (name + ('.jsonl' if name != 'summary' else '.json'))) == receipt[name + '_sha256']
assert digest(read(INPUT / 'plan/config.json')) == completed['config_hash']
assert digest(read(INPUT / 'plan/options.json')) == completed['options_hash']
plan = read(INPUT / 'plan/plan.json')
for name, expected in plan['inputs'].items():
    assert sha(INPUT / 'plan' / name) == expected, name
new, old = lines(INPUT / 'governed/episodes.jsonl'), lines(OLD / 'episodes.jsonl')
metrics = lines(INPUT / 'governed/metrics.jsonl')
report = read(INPUT / 'report/report.json')
assert report == read(ROOT / '3090结果/report.json')
assert len(new) == len(old) == len(metrics) == 6
assert [e['scenario_id'] for e in new] == completed['scenario_ids'] == [e['scenario_id'] for e in old]
for e, recorded, reported in zip(new, metrics, report['governed']):
    recomputed = metric(e, 400)
    assert all(same(recorded[k], v) for k, v in recomputed.items()), e['scenario_id']
    assert all(same(reported[k], v) for k, v in recorded.items()), e['scenario_id']
    assert len(e['path_option']['trace']) == e['terminal_step'] * 4
    assert not e['task_network_trained']
for e, recorded in zip(old, report['previous']):
    recomputed = metric(e, 400)
    assert all(same(recorded[k], v) for k, v in recomputed.items()), e['scenario_id']

scenes = read(INPUT / 'plan/development.json')['scenarios']
result = dict(completed=True, checks='All receipts, source, input hashes and 6 raw/reported metrics match',
              new_environment_initializations=0, training_updates=0, checkpoint_restore=False,
              linux_validation=validation, cases=[])
proximities, trajectories = [], []
for n, (s, o, e) in enumerate(zip(scenes, old, new)):
    # Exactly mirrors float32 initial position and float64 target integrator in
    # uav_env reset. Offline truth is used ONLY to audit logged trajectories.
    trajectory = simulate_target_trajectory(TargetState(np.asarray(s['target_initial_position'], np.float32),
        s['target_initial_velocity'], 0, s['target_motion_mode']), 400, .2, (20., 20., 8.),
        s['obstacles'], clearance=.2, max_reflections=4)
    trajectories.append(trajectory)
    prox = {label: proximity(raw, trajectory) for label, raw in (('previous', o), ('governed', e))}
    proximities.append(prox)
    for raw in (o, e):
        detected = [v['step'] for v in proximity(raw, trajectory) if v['margin'] <= 0]
        assert (min(detected) if detected else None) == raw['found_step']
    found, received = e['found_step'], e['executor_target_received_step']
    if found is not None:
        r = next(r for r in e['path_option']['trace'] if r['agent'] == 3 and r['step'] == found - 1)
        assert abs(dist(r['next_position'], trajectory[found].position) - e['executor_distance_to_target_at_found']) < 2e-6
    case = dict(scenario_id=e['scenario_id'], found_step=found, received_step=received,
                success_step=e['success_step'], timeout=e['termination_reason'] == 'timeout',
                previous_found_step=o['found_step'], previous_success_step=o['success_step'],
                closest_search={label: min(values, key=lambda x: x['margin']) for label, values in prox.items()},
                agents=[])
    for agent in range(4):
        rs = [r for r in e['path_option']['trace'] if r['agent'] == agent and r['phase'] != 3]
        slow = [r for r in rs if agent < 3 and np.linalg.norm(r['velocity']) < .05 and dist(r['endpoint'], r['position']) > .8]
        case['agents'].append(dict(agent=agent, active_steps=len(rs), reasons=Counter(r['reason'] for r in rs),
            actions=Counter(r['decision']['action'] for r in rs), slow_steps=len(slow),
            slow_actions=Counter(r['decision']['action'] for r in slow),
            motion_distance=sum(dist(r['next_position'], r['position']) for r in rs)))
    if received is not None:
        rs = [r for r in e['path_option']['trace'] if r['agent'] == 3 and r['phase'] == 2]
        ds = [(swept_relative_min_distance(r['position'], r['next_position'],
                trajectory[r['step']].position, trajectory[r['step']+1].position)[0], r['step']) for r in rs]
        case['capture'] = dict(contact_steps=e['capture_contact_step_count'], hold_max=e['capture_hold_counter_max'],
            closest_swept=dict(distance=min(ds)[0], starting_step=min(ds)[1]),
            brake_steps=sum(r['decision']['action'] == 8 for r in rs),
            blocked_steps=sum(r['navigation_blocked'] for r in rs),
            assignment_unreachable_count=e['executor_invalid_assignment_unreachable_count'],
            actual_handoff_distance=e['executor_distance_at_handoff'],
            target_prediction_error_summary=e['mean_target_prediction_error'],
            timeline=[dict(**row_brief(r), actual_target=trajectory[r['step']].position.tolist(),
                           actual_target_velocity=trajectory[r['step']].velocity.tolist())
                      for r in rs if r['step'] % 5 == 0 or r in (rs[0], rs[-1])])
    result['cases'].append(case)

rs = [r for r in new[2]['path_option']['trace'] if r['agent'] == 1 and r['step'] >= 100]
backtracks = []
for x, y in zip(rs, rs[1:]):
    if x['event_target'] == [5.,5.,2.5] and y['event_target'] == [3.,3.,2.5]:
        backtracks.append(dict(previous=row_brief(x), current=row_brief(y)))
result['case2_agent1'] = dict(window=[100,399], backward_tracking_resets=backtracks,
    unique_routes=len(set(r['route_id'] for r in rs)),
    tracking_targets=Counter(str(r['event_target']) for r in rs),
    xy_cap_median=float(np.median([r['action_diagnostics']['speed_cap_xy'] for r in rs])),
    total_slow_steps=result['cases'][2]['agents'][1]['slow_steps'])

rs = [r for r in new[4]['path_option']['trace'] if r['agent'] == 3 and r['step'] >= 378]
v = trajectories[4][399].velocity
capxy, capz = rs[-1]['action_diagnostics']['speed_cap_xy'], rs[-1]['action_diagnostics']['speed_cap_z']
offset = np.r_[2.4*v[:2]/capxy, 1.2*v[2]/capz]
result['case4_static_tracking'] = dict(window=[378,399], points=[row_brief(r) for r in rs],
    tracking_distance_min=min(r['distance_to_tracking'] for r in rs),
    tracking_distance_max=max(r['distance_to_tracking'] for r in rs),
    arrival_radius=.75, final_public_target_velocity=v.tolist(),
    simplified_zero_flow_steady_state_offset=offset.tolist(),
    simplified_offset_norm=float(np.linalg.norm(offset)),
    interpretation='Fixed waypoint with moving-target velocity feedforward admits nonzero stationary tracking error; simplified analytical counterexample, not a real-geometry rollout')
result['returns'] = {label: dict(discounted_mean=float(np.mean([e['team_discounted_return'] for e in data])),
    undiscounted_mean=float(np.mean([e['team_undiscounted_return'] for e in data]))) for label,data in (('previous',old),('governed',new))}
write('analysis.json', result)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, axes = plt.subplots(2,2, figsize=(12,8), constrained_layout=True)
ax=axes[0,0]
for label,color in (('previous','#d97706'),('governed','#2563eb')):
    values=proximities[1][label]
    grouped={t:min(v['margin'] for v in values if v['step']==t) for t in sorted(set(v['step'] for v in values))}
    ax.plot(list(grouped),list(grouped.values()),label=label,color=color,lw=1.4)
ax.axhline(0,color='black',ls='--',lw=1);ax.set(title='Case 0001: detection margin',ylabel='Distance minus sensor threshold (m)',xlabel='Physical step');ax.legend()
ax=axes[0,1]
rs=[r for r in new[2]['path_option']['trace'] if r['agent']==1]
ax.plot([r['step'] for r in rs],[r['distance_to_tracking'] for r in rs],color='#2563eb',lw=1.3,label='distance to path waypoint')
ax.axhline(.75,color='black',ls='--',lw=1,label='waypoint arrival radius')
for b in backtracks:ax.axvline(b['current']['step'],color='#dc2626',alpha=.3,lw=.8)
ax.set(title='Case 0002, agent 1: waypoint resets (red)',xlabel='Physical step',ylabel='Distance (m)');ax.legend(fontsize=8)
ax=axes[1,0]
rs=[r for r in new[4]['path_option']['trace'] if r['agent']==3 and r['phase']==2]
ax.plot([r['step'] for r in rs],[dist(r['position'],trajectories[4][r['step']].position) for r in rs],label='distance to actual target',color='#2563eb')
ax.axhline(.8,color='black',ls='--',label='capture radius');ax.set(title='Case 0004: after handoff',xlabel='Physical step',ylabel='Distance (m)');ax.legend(fontsize=8)
ax=axes[1,1]
ax.plot([r['step'] for r in rs],[r['distance_to_tracking'] for r in rs],label='distance to path waypoint',color='#2563eb')
ax.axhline(.75,color='black',ls='--',label='waypoint arrival radius');ax.axvspan(378,399,color='#f59e0b',alpha=.15)
ax.set(title='Case 0004: stalled near a fixed waypoint',xlabel='Physical step',ylabel='Distance (m)');ax.legend(fontsize=8)
fig.savefig(OUT/'diagnostics.png',dpi=160)
fig.savefig(OUT/'diagnostics.svg')
plt.close(fig)

after = snapshot_legacy()
input_after = {p.relative_to(ROOT).as_posix(): sha(p) for base in (INPUT,OLD)
               for p in sorted(base.rglob('*')) if p.is_file()}
assert after == before['legacy'] and input_after == before['inputs']
write('immutability.json', dict(legacy_unchanged=True,
      protected_counts={k:len(v) for k,v in after.items()}, uploaded_and_old_inputs_unchanged=True,
      environment_initializations=0, training_updates=0, checkpoint_restore=False))
write('created_files.json', [p.relative_to(ROOT).as_posix() for p in sorted(OUT.iterdir()) if p.is_file()] +
      [(OUT/'created_files.json').relative_to(ROOT).as_posix()])
print(json.dumps(dict(completed=True, cases=len(new), validation_tests=validation['synthetic']['tests'],
      physical_steps=sum(e['terminal_step'] for e in new), backwards_resets=len(backtracks),
      output=str(OUT), protected_counts={k:len(v) for k,v in after.items()}),ensure_ascii=False))
