"""Build an additive manifest without rewriting any parent manifest or evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from project_paths import ROOT
from chapter3_bser.path_option.source import digest, sha, legacy_digest, write


def build():
    target = ROOT/'provenance/path_option_manifest_v3.json'
    if target.exists():
        raise FileExistsError('manifest already frozen; make an explicit reviewed source revision')
    head = subprocess.check_output(['git', '-c', 'core.fsmonitor=false', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    tracked = subprocess.check_output(['git', '-c', 'core.fsmonitor=false', 'ls-tree', '-r', '--name-only', head], cwd=ROOT, text=True).splitlines()
    normalized = ['chapter3_bser/learning/provenance.py', 'tools/verify_source.py']
    parent = {}
    for name in tracked:
        if name.startswith(('core/', 'chapter3_bser/', 'provenance/', 'configs/')) or name == 'tools/verify_source.py':
            raw = subprocess.check_output(['git', 'show', f'{head}:{name}'], cwd=ROOT)
            if name in normalized:
                with tempfile.TemporaryDirectory(dir=ROOT/'work') as d:
                    p = Path(d)/Path(name).name; p.write_bytes(raw)
                    parent[name] = legacy_digest(p)
            else:
                parent[name] = hashlib.sha256(raw).hexdigest()
            actual = legacy_digest(ROOT/name) if name in normalized else sha(ROOT/name)
            if actual != parent[name]:
                raise ValueError('protected parent changed: '+name)
    additions = []
    additions += [p.relative_to(ROOT).as_posix() for p in (ROOT/'chapter3_bser/path_option').glob('*.py')]
    additions += [p.relative_to(ROOT).as_posix() for p in (ROOT/'tools').glob('*path_option*.py')]
    additions += ['checks/test_path_option.py', 'configs/path_option_v1.json', 'docs/PATH_OPTION_V1.md',
        'scripts/run_path_option_pilot.sh']
    files = dict(parent)
    files.update({n: sha(ROOT/n) for n in sorted(set(additions))})
    previous_path = ROOT/'provenance/path_option_manifest_v2.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8'))
    for version in (1, 2):
        name = f'provenance/path_option_manifest_v{version}.json'
        files[name] = sha(ROOT/name)
    archive_root = ROOT/'verification/path_option_v1'/previous['sha256']
    for archive_root in (ROOT/'verification/path_option_v1').iterdir():
        files.update({p.relative_to(ROOT).as_posix(): sha(p) for p in archive_root.iterdir() if p.is_file()})
    data = dict(schema='auv.path_option.source_manifest.v1', parent_commit=head,
        normalized_existing_dispatch=normalized, files=files,
        changes=[dict(path=n, before_sha256=None, after_sha256=files[n],
            reason='Independent OLD path-option source, tests, research interface or immutable draft evidence') for n in sorted(set(files)-set(parent))],
        parent_files=len(parent), parent_algorithm_bytes_changed=False,
        previous_candidate_sha256=previous['sha256'],
        draft_changes=[dict(path=n, before_sha256=previous['files'].get(n), after_sha256=files[n],
            reason='Align deployed categorical policy with value targets; separate learning evidence; retain incomplete evaluations')
            for n in sorted(set(additions)) if previous['files'].get(n) != files[n]],
        user_untracked_results_excluded=True, automatic_native_budget_remaining=0,
        budget_basis='Prior budget exhausted or unknown: no additional automatic environment initialization',
        preexisting_user_dispatch='Only two exact pure-MARL guard insertions normalize to parent AST; never staged')
    data['sha256'] = digest(data)
    write(target, data)
    return dict(manifest=str(target), source_sha256=data['sha256'], files=len(files))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--execute', action='store_true')
    a = p.parse_args(argv)
    print(json.dumps(build() if a.execute else dict(executed=False), indent=2))


if __name__ == '__main__':
    main()
