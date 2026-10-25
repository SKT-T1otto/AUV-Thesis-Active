"""Exact rebuilt-source identity and independently recorded manual readiness."""
import hashlib
import json
from pathlib import Path
from project_paths import ROOT

SCHEMA = 'auv.active_source.rebuild.v1'
GATES = ('source', 'structural', 'regression', 'parity', 'entrypoints', 'checkpoint')


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def source_files(root):
    root = Path(root).resolve()
    files = {}
    for folder in ('core', 'chapter3_bser', 'configs', 'checks', 'tools', 'provenance/parent', 'provenance/history'):
        for path in (root / folder).rglob('*'):
            if path.is_file() and path.suffix in ('.py', '.json'):
                files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('cli.py', 'project_paths.py', 'requirements.txt', 'AGENTS.md', '.gitattributes', '.gitignore', 'provenance/rebuild_evolution.json'):
        files[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    for path in (root/'provenance/baselines').rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.zip'):
            files[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    evolution = root/'provenance/map_stage1_evolution.json'
    if evolution.is_file():
        files[evolution.relative_to(root).as_posix()] = hashlib.sha256(evolution.read_bytes()).hexdigest()
    for name in ('provenance/map_stage1_manifest.json','provenance/map_core_evolution.json',
                 'provenance/map_core_manifest.json','provenance/map_experiment_evolution.json',
                 'provenance/map_experiment_manifest.json','provenance/map_convergence_evolution.json',
                 'provenance/map_convergence_manifest.json','provenance/map_persistence_evolution.json',
                 'provenance/map_cache_evolution.json','provenance/map_bounded_evolution.json','provenance/map_route_health_evolution.json','provenance/map_diagnostics_evolution.json'):
        if (root/name).is_file():files[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if (root/'provenance/map_cache_manifest.json').is_file():
        name='provenance/map_persistence_manifest.json'
        files[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if (root/'provenance/map_bounded_manifest.json').is_file():
        name='provenance/map_cache_manifest.json'
        files[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if (root/'provenance/map_route_health_manifest.json').is_file():
        name='provenance/map_bounded_manifest.json'
        files[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if (root/'provenance/map_diagnostics_manifest.json').is_file():
        name='provenance/map_route_health_manifest.json'
        files[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    for path in (root/'scripts').rglob('*.sh'):
        files[path.relative_to(root).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def verify_reviewed_changes(root=ROOT):
    root = Path(root).resolve()
    try:
        if (root/'provenance/map_diagnostics_manifest.json').is_file():
            from chapter3_bser.learning.map_diagnostic_source import verify
            return verify(root,source_files(root),canonical_digest)
        if (root/'provenance/map_route_health_manifest.json').is_file():
            from chapter3_bser.learning.map_route_health_source import verify
            return verify(root,source_files(root),canonical_digest)
        if (root/'provenance/map_bounded_manifest.json').is_file():
            from chapter3_bser.learning.map_bounded_source import verify
            return verify(root,source_files(root),canonical_digest)
        if (root/'provenance/map_cache_manifest.json').is_file():
            from chapter3_bser.learning.map_cache_source import verify
            return verify(root,source_files(root),canonical_digest)
        if (root/'provenance/map_persistence_manifest.json').is_file():
            from chapter3_bser.learning.map_persistence_source import verify
            return verify(root,source_files(root),canonical_digest)
        if (root/'provenance/map_convergence_manifest.json').is_file():
            from chapter3_bser.learning.map_convergence_source import verify
            return verify(root,source_files(root),canonical_digest)
        if (root/'provenance/map_experiment_manifest.json').is_file():
            from chapter3_bser.learning.map_experiment_source import verify
            return verify(root,source_files(root),canonical_digest)
        if (root/'provenance/map_core_manifest.json').is_file():
            from chapter3_bser.learning.map_core_source import verify
            return verify(root, source_files(root), canonical_digest)
        if (root/'provenance/map_stage1_manifest.json').is_file():
            from chapter3_bser.learning.map_source import verify
            return verify(root, source_files(root), canonical_digest)
        manifest = json.loads((root / 'provenance/active_manifest.json').read_text(encoding='utf-8'))
        body = {k: v for k, v in manifest.items() if k != 'sha256'}
        if manifest.get('schema') != SCHEMA or manifest.get('sha256') != canonical_digest(body):
            raise ValueError('invalid rebuild source manifest')
        actual = source_files(root)
        expected = manifest.get('files', {})
        if actual != expected:
            changed = sorted(k for k in actual.keys() | expected.keys() if actual.get(k) != expected.get(k))
            raise ValueError('active source/config inventory mismatch: ' + repr(changed))
        parent = json.loads((root / 'provenance/parent/source_receipt.json').read_text(encoding='utf-8'))
        if parent.get('source_verified') is not True or parent.get('source_identity') != manifest.get('parent_source_identity'):
            raise ValueError('unverified rebuild parent')
        history = json.loads((root / 'provenance/history/extraction_v1/active_manifest.json').read_text(encoding='utf-8'))
        evolution = json.loads((root / 'provenance/rebuild_evolution.json').read_text(encoding='utf-8'))
        if history.get('sha256') != manifest.get('previous_manifest_sha256') or evolution.get('previous_manifest_sha256') != history.get('sha256'):
            raise ValueError('rebuild lineage mismatch')
        changes = evolution.get('changes', [])
        indexed = {r['path']: r for r in changes}
        old = history['files']
        required = {p for p in old.keys() | actual.keys() if old.get(p) != actual.get(p)} - {'provenance/rebuild_evolution.json'}
        if len(indexed) != len(changes) or set(indexed) != required:
            raise ValueError('unreviewed rebuild evolution file set')
        for name, row in indexed.items():
            if row.get('before_sha256') != old.get(name) or row.get('after_sha256') != actual.get(name) or not row.get('reason'):
                raise ValueError('invalid rebuild evolution record: ' + name)
        return dict(review_sha256=manifest['sha256'], profile='active_rebuild_v1', inventory_sha256=canonical_digest(actual),
                    parent_commit=parent['head'], distribution=SCHEMA)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('rebuild source verification failed: ' + str(exc)) from exc


def require_review(identity):
    current = verify_reviewed_changes()
    if current != identity:
        raise ValueError('source identity changed during execution')
    return current


def research_readiness(identity, root=ROOT):
    root = Path(root).resolve()
    try:
        folder = {'active_map_diagnostics_v1':'verification/map_diagnostics_v1','active_map_route_health_v1':'verification/map_route_health_v1','active_map_bounded_v3':'verification/map_bounded_v3','active_map_cache_v1':'verification/map_cache_v1','active_map_persistence_v1':'verification/map_persistence_v1','active_map_convergence_v1':'verification/map_convergence_v1','active_map_experiment_v1':'verification/map_experiment_v1','active_map_core_v1':'verification/map_core_v1',
                  'active_map_interface_v1':'verification/map_stage1'}.get(identity.get('profile'),'verification/rebuild')
        if identity.get('profile') in ('active_map_persistence_v1','active_map_cache_v1','active_map_bounded_v3','active_map_route_health_v1','active_map_diagnostics_v1'):
            revision=root/folder/identity['review_sha256']
            if (revision/'receipt.json').is_file():folder=revision.relative_to(root).as_posix()
        receipt = json.loads((root / folder / 'receipt.json').read_text(encoding='utf-8'))
        gates = GATES + ('mechanisms',) if identity.get('profile') in ('active_map_core_v1','active_map_experiment_v1','active_map_convergence_v1','active_map_persistence_v1','active_map_cache_v1','active_map_bounded_v3','active_map_route_health_v1','active_map_diagnostics_v1') else GATES
        if (receipt.get('schema') != 'auv.active.rebuild.readiness.v1' or receipt.get('source_identity') != identity
                or receipt.get('manual_research_execution_allowed') is not True or receipt.get('performance_passed') is not False
                or receipt.get('historical_acceptance_transferred') is not False or set(receipt.get('gates', {})) != set(gates)):
            raise ValueError('manual research readiness is not established for this source')
        for gate, row in receipt['gates'].items():
            path = (root / row['path']).resolve()
            if not path.is_relative_to(root / folder) or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('missing or changed rebuild evidence: ' + gate)
            report = json.loads(path.read_text(encoding='utf-8'))
            if (report.get('gate') != gate or report.get('source_identity') != identity or report.get('passed') is not True
                    or report.get('exit_code') != 0 or report.get('run_id') != receipt['run_id']):
                raise ValueError('failed rebuild readiness gate: ' + gate)
        return receipt
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('manual research readiness evidence missing or invalid: ' + str(exc)) from exc


def training_admission(*, mechanism=False, config=None):
    identity = verify_reviewed_changes()
    if mechanism:
        if (config is None or config['max_steps'] > 32 or config['total_main_trajectories'] > 8
                or not 1 <= (config.get('max_total_environment_steps') or 0) <= 256):
            raise ValueError('mechanism execution requires <=32 steps, <=8 episodes and <=256 total start budget')
    else:
        research_readiness(identity)
    return identity
