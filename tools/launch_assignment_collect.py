"""Offline four-arm collector, with one reset and one departure ticket per arm."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import product
from pathlib import Path
import hashlib
import json
import random

import numpy as np
import torch

from project_paths import ROOT
from tools.launch_assignment_study import CONFIG, digest, file_hash, read, scene_identity
from tools.launch_intervention_source import checked_source


def add_arguments(parser):
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scene', default='r123_expand_20261119_0003')
    parser.add_argument('--repeat', type=int, default=0)
    parser.add_argument('--cap', type=int, help='external observation cap, defaults to native horizon')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=1)
    parser.add_argument('--reverse', action='store_true', help='reverse preview and submission traversal')
    parser.add_argument('--fixture', action='store_true', help='initialized actor for bounded mechanism checks')
    parser.add_argument('--execute', action='store_true')


def inputs(scene_id):
    study = read(CONFIG)
    native = study['native_inputs']
    config_path = ROOT / native['config']
    scenes = read(config_path.with_name('scenes.json'))['scenarios']
    scene, = [s for s in scenes if s['scenario_id'] == scene_id]
    physical = scene_identity(scene)['physical']
    family, = [f for f in read(ROOT / 'verification/launch_assignment_v1/manifest.json')['families']
               if any(m['physical'] == physical for m in f['members'])]
    if family['split'] != 'development':
        raise ValueError('collection requires a registered development family')
    return study, read(config_path), read(ROOT / native['options']), scene, family['family_id']


def initial_state(mission, scene):
    core = mission.core
    raw = core.unwrapped
    maps = [b.state.planning_graph.graph_sha256 for b in mission.boats]
    return dict(scene_digest=digest(scene), kinematics=asdict(core.get_agent_state()),
                task=asdict(core.get_task_state()), search=asdict(core.get_search_execution_state()),
                target=raw.target_state.to_payload(), acceleration=raw._agent_acc.tolist(),
                public_map_revision=int(raw.map_module.map_revision), local_graphs=maps,
                local_scans=[asdict(b.env.frame) for b in mission.boats],
                inboxes=[[asdict(r) for _, r in sorted(b.inbox.records.items())] for b in mission.boats],
                prior_task_ids=[b.lifecycle.task_id(b.agent) for b in mission.boats],
                standby=mission.boats[3].point.tolist(), packet=core.get_public_delivered_target(),
                random_python=digest(random.getstate()), random_torch=hashlib.sha256(torch.get_rng_state().numpy()).hexdigest(),
                random_numpy=digest([x.tolist() if isinstance(x, np.ndarray) else x for x in np.random.get_state()]))


def native_scalar(value):
    if isinstance(value, (np.generic, np.ndarray)):
        return value.tolist()
    raise TypeError(f'unexpected receipt field {type(value).__name__}')


def branch(job):
    from chapter3_bser.launch_assignment import LaunchAssignmentMission
    from chapter3_bser.joint_departure import JointDeparture
    from tools.check_launch_candidates import generators, make_actor, parameter_hash, write_receipt
    folder, req, arm = job
    path = folder / (arm + '.json')
    if path.exists():
        prior = read(path)
        if prior['request_digest'] != digest(req) or prior['arm'] != arm:
            raise ValueError('different existing branch: ' + arm)
        return arm
    torch.set_num_threads(1)
    study, config, options, scene, family = inputs(req['scene'])
    streams = study['random_streams']
    actor = make_actor(config['rl']['hidden_dim'], study['candidates']['proposer_reference'], not req['fixture'])
    continuation = generators(streams, family, 'continuation', req['repeat'])
    mission = LaunchAssignmentMission(config, options, scene, radius=study['problem']['information_boundary']['link_radius_m'],
        originals=generators(streams, family, 'original_offer'), extras=generators(streams, family, 'extra_offer'))
    ticket = JointDeparture([int(arm[0]), int(arm[1]), 0], req['order'])
    intervals = []

    def observe(m, cursor, task, delivery, result, team):
        event, = [e for e in m.core.read_public_events(cursor) if e.kind == 'step' and e.physical_step == task.step]
        acc = m.env.reward_accounting.last
        intervals.append(dict(step=task.step, event=asdict(event.payload), target=m.core.unwrapped.target_state.to_payload(),
                              packet=delivery, finder=int(task.finder_id), native_success=bool(task.mission_complete),
                              collision=bool(result['collision_episode']), team=team,
                              source_rewards=acc['source_reward_by_agent'], broadcast=acc['final_reward_by_agent']))
        if task.step % 100 == 0:
            print(f'{folder.name}/{arm} step {task.step}', flush=True)

    try:
        start = initial_state(mission, scene)
        raw = mission.core.unwrapped
        protocol = dict(ranges=raw._sensor_range[:3].tolist(), bias=float(raw.detect_eps_bias),
                        radius=raw.target_capture_radius, hold=raw.target_capture_hold_steps, horizon=mission.horizon)
        episode = mission.rollout(actor, continuation, req['cap'], departure=ticket, observe=observe)
        if not ticket.consumed:
            raise RuntimeError('departure was not consumed')
        result = dict(request_digest=digest(req), arm=arm, assignment=ticket.assignment, initial=start,
                      prefix=ticket.prefix, protocol=protocol, actor_sha256=parameter_hash(actor),
                      intervals=intervals, episode=episode,
                      native_discounted=mission.env.reward_accounting.team_discounted_return)
        write_receipt(path, json.loads(json.dumps(result, default=native_scalar, allow_nan=False)))
    finally:
        mission.close()
    return arm


def collect_command(args):
    from tools.check_launch_candidates import write_receipt
    study, config, _, scene, family = inputs(args.scene)
    cap = config['max_steps'] if args.cap is None else args.cap
    if not 1 <= cap <= config['max_steps'] or args.repeat < 0 or (args.fixture and cap > 32):
        raise ValueError('invalid observation cap or continuation repeat')
    req = dict(schema='auv.launch_intervention.v1', family=family, split='development', scene=args.scene,
               physical=scene_identity(scene)['physical'], repeat=args.repeat, cap=cap, fixture=args.fixture,
               continuation='unchanged_neighbor_policy', terminal='native_direct', gamma=.95,
               source_lock=checked_source(), order=list(reversed(range(4))) if args.reverse else list(range(4)),
               actor_reference=study['candidates']['proposer_reference'], assignment_axes=[0, 1], third_option=0)
    if not args.execute:
        print(json.dumps(req, ensure_ascii=False))
        return
    folder = args.output
    request = folder / 'request.json'
    if request.exists():
        if read(request) != req:
            raise ValueError('existing collection has a different request')
    else:
        write_receipt(request, req)
    jobs = [(folder, req, ''.join(bits)) for bits in product('01', repeat=2)]
    if args.workers == 1:
        for job in jobs:
            print('Collected', branch(job), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for arm in pool.map(branch, jobs):
                print('Collected', arm, flush=True)
    from tools.launch_assignment_audit import audit_folder
    summary = audit_folder(folder)
    out = folder / 'audit.json'
    if not out.exists():
        write_receipt(out, summary)
    elif read(out) != summary:
        raise ValueError('saved audit disagrees with reconstructed results')
    print(json.dumps(summary['counts']))
