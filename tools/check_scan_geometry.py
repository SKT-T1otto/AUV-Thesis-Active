"""Offline frozen-public-state checks; no environment, rollout or training."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
from project_paths import ROOT
from chapter3_bser.reach_avoid.control import ReachAvoidController
from chapter3_bser.reach_avoid.options import CONTRACT, SPECS, validate
from chapter3_bser.safe_rl.evidence import EvidenceGeometry
from chapter3_bser.safe_rl.dynamics import integrate, limits

FIXTURE = 'checks/fixtures/reach_avoid_0006_public_scans.json.gz'
FIXTURE_SHA256 = 'ccbfc54d2b634f678d11ef2a5d39b02118a6fb8d93979faf31c99cc21dc6c1d8'
BUFFER_FIXTURE = 'checks/fixtures/reach_avoid_buffer_public_scan.json'
BUFFER_SHA256 = '7bc36dbc71e36be7c454e349f3e846a3868774cc63f5eacd90922cd3404282f8'


def load_fixture():
    data = (ROOT / FIXTURE).read_bytes()
    if hashlib.sha256(data).hexdigest() != FIXTURE_SHA256:
        raise ValueError('public scan fixture identity changed')
    return json.loads(gzip.decompress(data))


def frozen_controller(fixture, step):
    row = next(x for x in fixture['snapshots'] if x['step'] == step)
    centers = (np.indices((10, 10, 8)).reshape(3, -1).T + .5) * [2., 2., 1.]
    masks = {}
    for key in ('occupied', 'free', 'unknown'):
        masks[key + '_mask'] = np.zeros(800, bool)
        masks[key + '_mask'][row[key]] = True
    state = NS(step=step, target_found=False, map_revision=row['map_revision'],
        grid=NS(shape=(10,10,8), origin=(0.,0.,0.), spacing=(2.,2.,1.), cell_centers=centers),
        occupancy=NS(**masks), agents=tuple(NS(agent_id=i, position=tuple(p), velocity=tuple(v))
            for i, (p,v) in enumerate(zip(row['positions'], row['velocities']))))
    events = [NS(sequence=s['step']+1, physical_step=s['step'], kind='scan', payload=NS(**{
        k: np.asarray(s[k]) for k in ('origins','directions','distances','hit_mask')}))
        for s in fixture['scans'] if s['step'] <= step]
    contract = dict(CONTRACT, bounds=(20.,20.,8.), obstacle_sensor_range=4.5,
        target_obstacle_clearance=.2, target_max_reflections_per_step=4)
    c = ReachAvoidController(SPECS, contract, validate(), 400)
    c.guard.evidence.consume(events[:-1])
    c.subcells.consume(events[:-1], step)
    c.guard.flow = np.asarray(row['flow'], float)
    packet = c.observe(state, NS(executor_knows_target=False), np.zeros((4,28)),
        dict(enumerate(row['semantic'])), events[-1:])
    return c, packet, state


def frozen_checks():
    fixture = load_fixture()
    records = []
    for step, agent in ((80,3), (171,0), (267,1)):
        c, packet, state = frozen_controller(fixture, step)
        p, v = packet.positions[agent], packet.velocities[agent]
        parent = EvidenceGeometry(state, c.guard.evidence, c.guard.options)
        if parent.raw.point_free(p) or parent.distance(p) != 0:
            raise ValueError('parent coarse-grid failure not reproduced')
        caps = c.guard._caps(p, v, SPECS[agent], c.geometry)
        actions, _, rows = c.act(packet)
        selected = rows[agent]
        moving = [j for j in np.flatnonzero(packet.checked[agent])
                  if np.linalg.norm(packet.actions[agent,j]-packet.actions[agent,8]) > 1e-4]
        if not caps[0] > 0 or not moving or selected['action_mode'] == 'UNCERTIFIED_BRAKE':
            raise ValueError(f'coarse-cell motion veto persists at {step}/{agent}')
        predicted_p, _ = integrate(p, v, actions[agent]*limits(SPECS[agent]), SPECS[agent], CONTRACT['dt'], packet.flow[agent])
        records.append(dict(step=step, agent_id=agent, parent_coarse_pose_free=False,
            ray_subcell_pose_free=bool(c.subcells.contains([p])[0]),
            speed_cap_xy=caps[0], speed_cap_z=caps[1],
            checked_moving_candidates=len(moving), selected_action=selected['action_mode'],
            selected_model_checked=selected['model_checked'],
            predicted_first_step_displacement=float(np.linalg.norm(predicted_p-p)),
            predicted_goal_distance_reduction=float(np.linalg.norm(packet.intents[agent]-p)-np.linalg.norm(packet.intents[agent]-predicted_p))))
    return dict(passed=True, fixture_sha256=FIXTURE_SHA256, records=records,
        near_buffer_initial=buffer_initial_checks(),
        environment_initializations=0, physical_steps=0, optimizer_updates=0,
        performance_passed=False, safety_guarantee=False,
        scope='Counterfactual action availability at recorded public states; no new trajectories or task-success claims')


def buffer_initial_checks():
    raw = (ROOT/BUFFER_FIXTURE).read_bytes()
    if hashlib.sha256(raw).hexdigest() != BUFFER_SHA256:
        raise ValueError('buffer public scan fixture changed')
    c, packet, state = frozen_controller(json.loads(raw), 0)
    _, _, rows = c.act(packet)
    if any(r['action_mode']!='BUFFER_EXIT' or not r['model_checked'] for r in rows):
        raise ValueError('near-buffer public initial state lost its checked outward exit')
    return dict(passed=True, fixture_sha256=BUFFER_SHA256,
        roles=[dict(agent_id=r['agent_id'], action_mode=r['action_mode'], model_checked=r['model_checked']) for r in rows],
        environment_initializations=0, physical_steps=0,
        scope='Initial public observation only, finite model exit probes; no environment trajectory')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args(argv)
    if not a.execute:
        print(json.dumps(dict(executed=False, output=a.output)))
        return
    from chapter3_bser.learning.provenance import verify_reviewed_changes, require_review
    source = verify_reviewed_changes()
    output = Path(a.output).resolve()
    if not output.is_relative_to(ROOT/'work'):
        raise ValueError('fresh output under work/ required')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(frozen_checks(), source=source)
    require_review(source)
    with (output/'scan_geometry.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
