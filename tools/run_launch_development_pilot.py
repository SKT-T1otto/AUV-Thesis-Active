"""Prepare or explicitly run the preregistered three-family launch assignment pilot."""
import argparse
import copy
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from chapter3_bser.launch_assignment import LaunchAssignmentMission
from project_paths import ROOT
from tools.check_launch_candidates import generators, make_actor, write_receipt
from tools.launch_assignment_audit import audit_folder, require
from tools.launch_assignment_collect import collect_command, inputs, initial_state, native_scalar
from tools.launch_assignment_study import CONFIG, digest, file_hash, read
from tools.launch_development_pilot_audit import OUT, REGISTRY, departure_actions, family_prefix, publish_summary
from tools.launch_intervention_source import checked_source


PLAN_HASH = '7e64707eb07d6f85bf0001a4036ebaf3f4a4d873606b9a027da43271002f14f0'


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


def cells(plan):
    for family in plan['families']:
        scene = family['scene']
        for repeat in plan['repeats']:
            archived = family == plan['families'][0] and repeat == 0
            folder = ROOT / plan['baseline'] if archived else OUT / f"{scene.rsplit('_', 1)[1]}_r{repeat}"
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
        menus = [asdict(boat.preview(copy.deepcopy(actor).eval())) for boat in mission.boats[:3]]
        if family == plan['families'][0]:
            ref = read(ROOT / plan['baseline'] / '00.json')
            serialized = json.loads(json.dumps(initial, default=native_scalar, allow_nan=False))
            require(digest(serialized) == digest(ref['initial']), 'historical 0003 reset differs on this runtime')
            require(digest(menus) == digest(ref['prefix']['menus']), 'historical 0003 frozen menus differ on this runtime')
        sizes = [len(menu['points']) for menu in menus]
        return dict(scene=family['scene'], family=family_id, menu_sizes=sizes, menus=menus,
                    four_arms_available=sizes[0] >= 2 and sizes[1] >= 2 and sizes[2] >= 1)
    finally:
        mission.close()


def run(plan, workers):
    torch.set_num_threads(1)
    missing = []
    for family in plan['families']:
        menu = preview(plan, family)
        scene = family['scene']
        if not menu['four_arms_available']:
            # A missing option precedes assignment. A subsequent native rejection is still an assigned arm.
            path = OUT / (scene + '_unavailable.json')
            if path.exists():
                require(digest(read(path)) == digest(menu), 'different saved availability receipt')
            else:
                write_receipt(path, menu)
            print(f"{scene}: cannot issue four assignments; frozen menu sizes {menu['menu_sizes']}", flush=True)
            missing.append(scene)
            continue
        reference = None
        for scheduled, repeat, archived, folder in cells(plan):
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
            identity = family_prefix(records[0]), [departure_actions(b) for b in records]
            if reference is None:
                reference = identity
            require(identity == reference, f'{scene} repeat {repeat}: reset, candidates or departure changed')
    if missing:
        raise SystemExit('Incomplete pilot; four-way assignment unavailable in ' + ', '.join(missing) +
                         '. Other eligible blocks retain their audit.json. No substitute family or full-pilot verdict.')
    publish_summary(OUT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--pack-inputs', type=Path, metavar='ZIP', help='export the seven frozen inputs and exit')
    group.add_argument('--inputs', type=Path, metavar='ZIP', help='verify and import the portable inputs if needed')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=2)
    parser.add_argument('--execute', action='store_true', help='run missing development branches')
    args = parser.parse_args()
    if args.execute and args.pack_inputs:
        parser.error('--pack-inputs is a separate export operation')
    require(file_hash(OUT / 'plan.json') == PLAN_HASH, 'different pilot plan')
    plan = read(OUT / 'plan.json')
    require(file_hash(REGISTRY) == plan['registry_sha256'] and checked_source() == plan['source_lock'],
            'different frozen registry or runtime')
    expected = portable_inputs(plan)
    if args.pack_inputs:
        pack(args.pack_inputs, expected)
        print(f'{file_hash(args.pack_inputs)}  {args.pack_inputs}')
        return
    if args.inputs:
        with exclusive_run(OUT):
            install(args.inputs, expected)
    for name, checksum in expected.items():
        path = ROOT / name
        require(path.is_file(), 'missing frozen input; supply --inputs ZIP: ' + name)
        require(file_hash(path) == checksum, 'incorrect frozen input: ' + name)
    baseline = ROOT / plan['baseline']
    require(read(baseline / 'audit.json') == audit_folder(baseline), 'archived baseline audit differs')
    pending = sum(not (folder / (arm + '.json')).exists() for _, _, _, folder in cells(plan) for arm in plan['arms'])
    print(f"Planned {len(plan['families']) * len(plan['repeats']) * len(plan['arms'])} terminals, "
          f"4 archived, {pending} branch receipts missing.", flush=True)
    if args.execute:
        with exclusive_run(OUT):
            run(plan, args.workers)


if __name__ == '__main__':
    main()
