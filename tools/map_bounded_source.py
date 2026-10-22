"""Independent v3 evolution verification; frozen v2 bytes are data, never code."""
import hashlib
import json
from pathlib import Path
import zipfile
from tools.map_interface_source import digest,legacy_ast
from tools.map_cache_source import inventory as cache_inventory

MANIFEST='provenance/map_bounded_manifest.json'
EVOLUTION='provenance/map_bounded_evolution.json'
PARENT='provenance/map_cache_manifest.json'
BASELINE='provenance/baselines/map_cache_20261022_v1'


def inventory(root):
    root=Path(root);actual=cache_inventory(root)
    for name in (PARENT,EVOLUTION):actual[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    return actual


def verify(root):
    root=Path(root);read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    sha=lambda n:hashlib.sha256((root/n).read_bytes()).hexdigest()
    data,parent,base=read(MANIFEST),read(PARENT),read(BASELINE+'/baseline.json');actual=inventory(root)
    if data.get('schema')!='auv.active_source.map_bounded.v1' or data.get('sha256')!=digest({k:v for k,v in data.items() if k!='sha256'}):raise ValueError('invalid bounded manifest')
    if actual!=data['files']:raise ValueError('independent source inventory mismatch')
    if (parent['sha256']!=digest({k:v for k,v in parent.items() if k!='sha256'})
            or data['parent_review_sha256']!=parent['sha256'] or data['parent_manifest_sha256']!=sha(PARENT)
            or base['source_identity']['review_sha256']!=parent['sha256']
            or base['archive_sha256']!=sha(BASELINE+'/source_and_evidence.zip')):raise ValueError('immutable cache parent changed')
    external=base['external_frozen_archives']
    with zipfile.ZipFile(root/BASELINE/'source_and_evidence.zip') as archive:
        if set(archive.namelist())!=set(base['files'])-set(external):raise ValueError('parent archive inventory changed')
        for name,expected in base['files'].items():
            frozen=sha(name) if name in external else hashlib.sha256(archive.read(name)).hexdigest()
            if frozen!=expected or (name in external and external[name]!=expected):raise ValueError('frozen parent member changed')
            if name.startswith(('provenance/','verification/')) and sha(name)!=expected:raise ValueError('parent evidence changed: '+name)
        for name,expected in parent['files'].items():
            if base['files'].get(name)!=expected:raise ValueError('frozen parent source mismatch')
        for name in ('core/env/uav_env.py','core/env/mission_env.py','chapter3_bser/learning/base_runtime.py'):
            if legacy_ast(archive.read(name).decode())!=legacy_ast((root/name).read_text(encoding='utf-8')):raise ValueError('legacy AST changed: '+name)
        for name in ('geometry_hypotheses.py','geometry_cache.py','coupled_belief.py','target_particles.py'):
            path='chapter3_bser/experiments/coupled_search_v1/'+name
            if (root/path).read_bytes()!=archive.read(path):raise ValueError('protected geometry kernel changed: '+name)
    permitted={
        'cli.py','chapter3_bser/experiments/coupled_search_v1/config.py',
        'chapter3_bser/experiments/coupled_search_v1/core_config.py',
        'chapter3_bser/experiments/coupled_search_v1/reference.py',
        'chapter3_bser/experiments/coupled_search_v1/session.py',
        'chapter3_bser/learning/map_contract.py','chapter3_bser/learning/map_suite.py',
        'chapter3_bser/learning/provenance.py','checks/test_map_source.py',
        'configs/rebuild_check_contract.json','tools/verify_source.py','tools/validate_checkout.py',
        'tools/check_map_training.py','tools/run_checks.py'}
    changed={n for n,h in parent['files'].items() if actual.get(n)!=h}
    if changed-permitted:raise ValueError('protected algorithm source changed: '+repr(sorted(changed-permitted)))
    new_runtime={n for n in actual.keys()-parent['files'].keys() if n.startswith(('core/','chapter3_bser/'))}
    expected={'chapter3_bser/experiments/coupled_search_v1/'+n for n in ('bounded_belief.py','bounded_config.py','persistent_map.py')}
    expected|={'chapter3_bser/learning/'+n for n in ('bounded_experiment.py','map_bounded_source.py')}
    if new_runtime!=expected:raise ValueError('unexpected runtime extension')
    evolution=read(EVOLUTION);rows=evolution['changes'];indexed={r['path']:r for r in rows}
    required={n for n in parent['files'].keys()|actual.keys() if parent['files'].get(n)!=actual.get(n)}-{EVOLUTION}
    if len(indexed)!=len(rows) or set(indexed)!=required or evolution['parent_review_sha256']!=parent['sha256']:raise ValueError('bounded evolution coverage mismatch')
    for name,row in indexed.items():
        if row.get('before_sha256')!=parent['files'].get(name) or row.get('after_sha256')!=actual.get(name) or not row.get('reason'):raise ValueError('bounded evolution hash mismatch: '+name)
    return dict(passed=True,files=len(actual),manifest_sha256=data['sha256'],map_algorithm_revision='coupled_search_bounded_v3',
        algorithm_changed=True,learner_and_physics_byte_identical=True,all_parent_evidence_immutable=True,
        baseline_archive_verified=True,original_v2_byte_identical=True)
