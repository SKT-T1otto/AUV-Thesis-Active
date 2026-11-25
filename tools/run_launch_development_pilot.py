"""Prepare or explicitly run the preregistered three-family launch assignment pilot."""
import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import torch
import numpy as np

from chapter3_bser.launch_assignment import LaunchAssignmentMission
from chapter3_bser.joint_departure import stream_states
from project_paths import ROOT
from tools.check_launch_candidates import generators, make_actor, write_receipt
from tools.launch_assignment_audit import audit_folder, require
from tools.launch_assignment_collect import collect_command, inputs, initial_state, native_scalar
from tools.launch_assignment_study import CONFIG, digest, file_hash, read
from tools.launch_development_pilot_audit import OUT, REGISTRY, departure_actions, family_prefix, publish_summary
from tools.launch_intervention_source import checked_source


PLAN_HASH = '7e64707eb07d6f85bf0001a4036ebaf3f4a4d873606b9a027da43271002f14f0'
LOCAL_OUT = ROOT / 'verification/launch_development_pilot_v2'


def portable_inputs(plan):
    actor = read(CONFIG)['candidates']['proposer_reference']
    require(actor['file_sha256'] == plan['checkpoint_identity'] and
            actor['actor_parameters_sha256'] == plan['actor_sha256'], 'checkpoint identity differs')
    names = {f"{plan['baseline']}/{name}": h for name, h in plan['baseline_sha256'].items()}
    names[actor['path']] = actor['file_sha256']
    return names


def publish_file(staged, path):
    if os.name == 'nt':
        os.rename(staged, path)
    else:
        os.link(staged, path)


def check_archive(archive, expected):
    with ZipFile(archive) as packed:
        require(sorted(packed.namelist()) == sorted(expected), 'input archive members differ')
        for name, checksum in expected.items():
            require(hashlib.sha256(packed.read(name)).hexdigest() == checksum, 'input archive mismatch: ' + name)


def pack(archive, expected):
    if archive.exists():
        check_archive(archive, expected)
        return
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=archive.parent) as temp:
        staged = Path(temp) / 'inputs.zip'
        with ZipFile(staged, 'w', ZIP_DEFLATED) as packed:
            for name in expected:
                packed.write(ROOT / name, name)
        check_archive(staged, expected)
        publish_file(staged, archive)


def install(archive, expected, root=ROOT):
    check_archive(archive, expected)
    pending = []
    for name, checksum in expected.items():
        path = root / name
        if path.exists():
            require(file_hash(path) == checksum, 'conflicting existing input: ' + name)
        else:
            pending.append(name)
    with ZipFile(archive) as packed:
        for name in pending:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=path.parent) as temp:
                staged = Path(temp) / 'input'
                staged.write_bytes(packed.read(name))
                require(file_hash(staged) == expected[name], 'archive changed during import: ' + name)
                publish_file(staged, path)


@contextmanager
def exclusive_run(out):
    # The lock follows the open descriptor and releases on exit or interruption.
    with (out / 'pilot.lock').open('a+b') as handle:
        try:
            if os.name == 'nt':
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError('the pilot directory is locked by another run') from exc
        yield


def cells(plan, out=None):
    out = OUT if out is None else out
    for family in plan['families']:
        scene = family['scene']
        for repeat in plan['repeats']:
            archived = plan.get('reuse_archived_baseline', True) and family == plan['families'][0] and repeat == 0
            folder = ROOT / plan['baseline'] if archived else out / f"{scene.rsplit('_', 1)[1]}_r{repeat}"
            yield scene, repeat, archived, folder


def preview(plan, family):
    """Recreate the native reset and private draws without issuing an assignment or taking a step."""
    study, config, options, scene, family_id = inputs(family['scene'])
    require(family_id == family['family'], 'different registered family')
    actor = make_actor(config['rl']['hidden_dim'], study['candidates']['proposer_reference'], True)
    streams = study['random_streams']
    mission = LaunchAssignmentMission(config, options, scene,
        radius=study['problem']['information_boundary']['link_radius_m'],
        originals=generators(streams, family_id, 'original_offer'), extras=generators(streams, family_id, 'extra_offer'))
    try:
        initial = initial_state(mission, scene)
        continuation = generators(streams, family_id, 'continuation')
        before = stream_states(mission.boats, continuation)
        menus = [asdict(boat.preview(copy.deepcopy(actor).eval())) for boat in mission.boats[:3]]
        serialized = json.loads(json.dumps(initial, default=native_scalar, allow_nan=False))
        raw = mission.core.unwrapped
        snapshot = dict(initial=serialized, actor_sha256=plan['actor_sha256'],
                        protocol=dict(ranges=raw._sensor_range[:3].tolist(), bias=float(raw.detect_eps_bias),
                                      radius=raw.target_capture_radius, hold=raw.target_capture_hold_steps,
                                      horizon=mission.horizon),
                        prefix=dict(menus=menus, streams_before=before,
                                    streams_after=stream_states(mission.boats, continuation),
                                    draws=[[asdict(draw) for draw in boat._draws] for boat in mission.boats[:3]]))
        comparison = None
        if family == plan['families'][0]:
            ref = read(ROOT / plan['baseline'] / '00.json')
            comparison = historical_comparison(snapshot, ref)
            if plan.get('reuse_archived_baseline', True):
                require(comparison['initial_equal'], 'historical 0003 reset differs on this runtime')
                require(comparison['menus_equal'], 'historical 0003 frozen menus differ on this runtime')
        sizes = [len(menu['points']) for menu in menus]
        return dict(scene=family['scene'], family=family_id, menu_sizes=sizes, menus=menus,
                    snapshot=snapshot, prefix_sha256=family_prefix(snapshot), historical_comparison=comparison,
                    four_arms_available=sizes[0] >= 2 and sizes[1] >= 2 and sizes[2] >= 1)
    finally:
        mission.close()


def historical_comparison(snapshot, reference):
    old = reference['prefix']['menus']
    new = snapshot['prefix']['menus']
    differences = []
    for a, b in zip(new, old):
        coordinates = np.asarray(a['points'], dtype=float)
        archived = np.asarray(b['points'], dtype=float)
        same_shape = coordinates.shape == archived.shape
        differences.append(dict(boat=a['boat'], same_shape=same_shape,
            features_equal=digest(a['features']) == digest(b['features']),
            max_coordinate_difference_m=float(np.max(np.abs(coordinates - archived), initial=0.)) if same_shape else None))
    return dict(initial_equal=digest(snapshot['initial']) == digest(reference['initial']),
                menus_equal=digest(new) == digest(old), coordinates=differences)


def runtime_identity():
    return dict(system=platform.system(), machine=platform.machine(), python=sys.version,
                numpy=np.__version__, torch=str(torch.__version__),
                torch_build_sha256=digest(torch.__config__.show()),
                numpy_build_sha256=digest(np.show_config(mode='dicts')),
                cpu_capability=torch.backends.cpu.get_cpu_capability(), torch_threads=torch.get_num_threads(),
                environment={key: os.environ.get(key) for key in
                             ('MKL_CBWR', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'ATEN_CPU_CAPABILITY')})


def freeze_runtime(plan, out):
    torch.set_num_threads(1)
    receipt = dict(schema='auv.launch_development_pilot.runtime.v2',
                   plan_sha256=file_hash(out / 'plan.json'), source_lock=checked_source(),
                   orchestration_sha256={p: file_hash(ROOT / p) for p in
                       ('tools/run_launch_development_pilot.py', 'tools/launch_development_pilot_audit.py')},
                   runtime=runtime_identity(), departures=[])
    saved = out / 'runtime.json'
    previous = read(saved) if saved.exists() else None
    if previous:
        require({k: v for k, v in previous.items() if k != 'departures'} ==
                {k: v for k, v in receipt.items() if k != 'departures'}, 'runtime or orchestration changed since preparation')
    for family in plan['families']:
        print(f"Checking frozen departure {family['scene']} (zero physical steps)", flush=True)
        receipt['departures'].append(preview(plan, family))
    if previous:
        require(digest(previous) == digest(receipt), 'local frozen departures changed since preparation')
    else:
        # A runtime receipt must predate every collected terminal.
        require(not any(out.glob('*_r*/request.json')), 'existing branches have no runtime receipt')
        write_receipt(saved, receipt)
    print('Runtime departures verified. Historical terminals are retained as reference.', flush=True)
    return {item['scene']: item for item in receipt['departures']}


def run(plan, workers, out=None, previews=None):
    out = OUT if out is None else out
    torch.set_num_threads(1)
    missing = []
    for family in plan['families']:
        menu = preview(plan, family) if previews is None else previews[family['scene']]
        scene = family['scene']
        if not menu['four_arms_available']:
            # A missing option precedes assignment. A subsequent native rejection is still an assigned arm.
            path = out / (scene + '_unavailable.json')
            if path.exists():
                require(digest(read(path)) == digest(menu), 'different saved availability receipt')
            else:
                write_receipt(path, menu)
            print(f"{scene}: cannot issue four assignments; frozen menu sizes {menu['menu_sizes']}", flush=True)
            missing.append(scene)
            continue
        reference = None
        for scheduled, repeat, archived, folder in cells(plan, out):
            if scheduled != scene:
                continue
            print(f"{scene} repeat {repeat}: {'archived' if archived else 'collect/resume'}", flush=True)
            if not archived:
                collect_command(argparse.Namespace(scene=scene, repeat=repeat, cap=plan['cap'], output=folder,
                                workers=workers, fixture=False, reverse=False, execute=True))
            stored = read(folder / 'audit.json')
            require(stored['counts']['native_terminal'] == len(plan['arms']), 'incomplete terminal block')
            records = [read(folder / (arm + '.json')) for arm in plan['arms']]
            require(digest(records[0]['prefix']['menus']) == digest(menu['menus']), 'collected menu differs from preview')
            if not plan.get('reuse_archived_baseline', True):
                require(all(family_prefix(b) == menu['prefix_sha256'] for b in records),
                        'collected departure differs from the runtime receipt')
            identity = family_prefix(records[0]), [departure_actions(b) for b in records]
            if reference is None:
                reference = identity
            require(identity == reference, f'{scene} repeat {repeat}: reset, candidates or departure changed')
    if missing:
        raise SystemExit('Incomplete pilot; four-way assignment unavailable in ' + ', '.join(missing) +
                         '. Other eligible blocks retain their audit.json. No substitute family or full-pilot verdict.')
    publish_summary(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pack-inputs', type=Path, metavar='ZIP', help='export the seven frozen inputs and exit')
    group.add_argument('--inputs', type=Path, metavar='ZIP', help='verify and import the portable inputs if needed')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=2)
    parser.add_argument('--runtime-local', action='store_true', help='v2: collect all 48 terminals in this runtime')
    parser.add_argument('--execute', action='store_true', help='run missing development branches')
    args = parser.parse_args()
    if args.execute and args.pack_inputs:
        parser.error('--pack-inputs is a separate export operation')
    require(file_hash(OUT / 'plan.json') == PLAN_HASH, 'different pilot plan')
    plan = read(OUT / 'plan.json')
    out = OUT
    if args.runtime_local:
        out = LOCAL_OUT
        local_plan = read(out / 'plan.json')
        expected = dict(plan, schema='auv.launch_development_pilot.v2', parent_plan_sha256=PLAN_HASH,
                        amended_date='2026-11-25',
                        reuse_archived_baseline=False, native_episodes_new=48, max_new_physical_steps=19200,
                        amendment='Recollect repeat zero in the collection runtime after observed cross-platform coordinate differences.')
        require(local_plan == expected, 'different runtime-local plan')
        plan = local_plan
    require(file_hash(REGISTRY) == plan['registry_sha256'] and checked_source() == plan['source_lock'],
            'different frozen registry or runtime')
    expected = portable_inputs(plan)
    if args.pack_inputs:
        pack(args.pack_inputs, expected)
        print(f'{file_hash(args.pack_inputs)}  {args.pack_inputs}')
        return
    if args.inputs:
        with exclusive_run(out):
            install(args.inputs, expected)
    for name, checksum in expected.items():
        path = ROOT / name
        require(path.is_file(), 'missing frozen input; supply --inputs ZIP: ' + name)
        require(file_hash(path) == checksum, 'incorrect frozen input: ' + name)
    baseline = ROOT / plan['baseline']
    require(read(baseline / 'audit.json') == audit_folder(baseline), 'archived baseline audit differs')
    pending = sum(not (folder / (arm + '.json')).exists() for _, _, _, folder in cells(plan, out) for arm in plan['arms'])
    reused = 0 if args.runtime_local else 4
    print(f"Planned {len(plan['families']) * len(plan['repeats']) * len(plan['arms'])} terminals, "
          f"{reused} archived terminals pooled, {pending} branch receipts missing.", flush=True)
    if args.execute or args.runtime_local:
        with exclusive_run(out):
            previews = freeze_runtime(plan, out) if args.runtime_local else None
            if args.execute:
                run(plan, args.workers, out, previews)


if __name__ == '__main__':
    main()
