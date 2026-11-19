"""Seal preparation evidence and deliver new files; no full evaluation."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tools.r123_expansion import identity, load_plan


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    (OUT / name).write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


def main():
    source = identity()
    plan, config, options, scenes = load_plan(OUT / 'plan')
    checks = read(OUT / 'windows/checks.json')
    assert checks['passed'] and checks['tool'] == source and checks['synthetic']['tests'] == 17
    assert checks['observer_trace_equivalence'] and checks['environment_initializations'] == 4 and checks['physical_steps'] <= 128
    assert checks['training_updates'] == 0 and not checks['checkpoint_load']
    assert checks['plan_sha256'] == sha(OUT / 'plan/plan.json')
    for name, h in checks['artifacts'].items():
        assert sha(OUT / 'windows' / name) == h
    raw = [read(OUT / f'windows/episode_{i}.json') for i in range(4)]
    assert raw[0]['path_option']['trace'] == raw[1]['path_option']['trace']
    assert raw[2]['path_option']['trace'] == raw[3]['path_option']['trace']
    assert not (OUT / 'dry_run_must_not_exist').exists()
    assert b'\r' not in (ROOT / 'scripts/r123_expansion_linux.sh').read_bytes()
    write('validation_summary.json', dict(passed=True, tool=source, new_scenes=24, planned_full_episodes=48,
        full_linux_evaluation_started=False, maximum_linux_evaluation_steps=19200,
        synthetic_tests=17, environment_initializations=4, windows_physical_steps=checks['physical_steps'],
        training_updates=0, checkpoint_load=False, observer_physical_trace_equivalence=True,
        short_collisions=[read(OUT / f'windows/metric_{i}.json')['collision'] for i in range(4)],
        dry_run_creates_no_output=True, shell_file_has_unix_line_endings=True))
    before = read(ROOT / 'verification/r123_global_result_audit/20261119_v1/before.json')
    write('before.json', before)
    for i, (name, record) in enumerate(before.items()):
        p = ROOT.parent / name
        assert p.stat().st_size == record['bytes'] and sha(p) == record['sha256'], name
        if i % 250 == 0:
            print(f'[unchanged parent/input] {i}/{len(before)}', flush=True)
    write('immutability.json', dict(parent_and_input_files_unchanged=len(before),
        protected_counts={n: sum(k.startswith(n + '/') for k in before) for n in ('CH3', 'CH4', 'CH5')},
        source_unchanged=True, historical_results_unchanged=True, protected_byte_identical=True,
        new_evaluation_physical_steps=0, windows_interface_check_steps=checks['physical_steps']))
    new_files = list(read(ROOT / 'provenance/r123_expansion_manifest_v1.json')['files'])
    new_files.append('provenance/r123_expansion_manifest_v1.json')
    package = OUT / 'r123_expansion_new_files.zip'
    with zipfile.ZipFile(package, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for name in new_files:
            bundle.write(ROOT / name, arcname=name)
    with zipfile.ZipFile(package) as bundle:
        assert set(bundle.namelist()) == set(new_files)
        for name in bundle.namelist():
            assert hashlib.sha256(bundle.read(name)).hexdigest() == sha(ROOT / name)
    write('delivery.json', dict(package=package.relative_to(ROOT).as_posix(), sha256=sha(package),
        files={name: sha(ROOT / name) for name in new_files},
        github_pushed=False, notice='Only nine newly named source/data files. Parent control and historical results are not in this archive.'))
    created = {p.relative_to(ROOT).as_posix() for p in OUT.rglob('*') if p.is_file()}
    created.update(new_files)
    created.update(p.relative_to(ROOT).as_posix() for base, prefix in [('tools', 'r123_expansion.'), ('checks', 'test_r123_expansion.')]
        for p in (ROOT / base / '__pycache__').glob(prefix + '*.pyc'))
    created.update((OUT / name).relative_to(ROOT).as_posix() for name in ('created_files.json', 'completed.json'))
    write('created_files.json', sorted(created))
    write('completed.json', dict(preparation_completed=True, evaluation_completed=False, tool=source,
        new_scenes=24, planned_paired_jobs=48, maximum_linux_physical_steps=19200,
        windows_physical_steps=checks['physical_steps'], training_updates=0, checkpoint_loads=0,
        artifacts={p.relative_to(OUT).as_posix(): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'completed.json'}))
    print(json.dumps(dict(preparation_completed=True, full_evaluation_started=False, new_files=9,
        windows_steps=checks['physical_steps'], package=str(package))))


if __name__ == '__main__':
    main()
