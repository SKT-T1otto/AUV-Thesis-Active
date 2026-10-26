"""Independent exact-source successor checks, without importing runtime code."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile
from tools.map_diagnostic_source import inventory as parent_inventory
from tools.map_interface_source import digest, legacy_ast

MANIFEST='provenance/executor_standby_manifest.json'
EVOLUTION='provenance/executor_standby_evolution.json'
PARENT='provenance/map_diagnostics_manifest.json'
BASELINE='provenance/baselines/map_diagnostics_20261026_v1'


def inventory(root):
    root=Path(root);actual=parent_inventory(root)
    for n in (PARENT,EVOLUTION):actual[n]=hashlib.sha256((root/n).read_bytes()).hexdigest()
    return actual


def verify(root):
    root=Path(root);read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    sha=lambda n:hashlib.sha256((root/n).read_bytes()).hexdigest()
    actual=inventory(root);data,parent,base=read(MANIFEST),read(PARENT),read(BASELINE+'/baseline.json')
    if data.get('schema')!='auv.active_source.executor_standby.v1' or data.get('sha256')!=digest({k:v for k,v in data.items() if k!='sha256'}):raise ValueError('invalid executor standby manifest')
    if actual!=data['files']:raise ValueError('independent source inventory mismatch')
    if (data['parent_review_sha256']!=parent['sha256'] or data['parent_manifest_sha256']!=sha(PARENT)
            or base['source_identity']['review_sha256']!=parent['sha256'] or sha(BASELINE+'/source_and_evidence.zip')!=base['archive_sha256']):
        raise ValueError('immutable diagnostics parent changed')
    allowed={'chapter3_bser/planner_factory.py','chapter3_bser/learning/map_contract.py',
             'chapter3_bser/learning/map_suite.py','chapter3_bser/learning/provenance.py'}
    with zipfile.ZipFile(root/BASELINE/'source_and_evidence.zip') as archive:
        external=base['external_frozen_archives']
        if set(archive.namelist())!=set(base['files'])-set(external):raise ValueError('parent archive inventory changed')
        for n,h in base['files'].items():
            saved=sha(n) if n in external else hashlib.sha256(archive.read(n)).hexdigest()
            if saved!=h:raise ValueError('frozen parent member changed: '+n)
            if n.startswith(('provenance/','verification/')) and sha(n)!=h:raise ValueError('parent evidence changed: '+n)
        for n in ('core/env/uav_env.py','core/env/mission_env.py','chapter3_bser/learning/base_runtime.py'):
            if legacy_ast(archive.read(n).decode())!=legacy_ast((root/n).read_text(encoding='utf-8')):raise ValueError('legacy AST changed: '+n)
        for n,h in parent['files'].items():
            if n.startswith(('core/','chapter3_bser/')) and n not in allowed and actual.get(n)!=h:
                if 'geometry_hypotheses.py' in n:raise ValueError('protected geometry kernel changed: '+n)
                if 'bounded_belief.py' in n:raise ValueError('observer changed algorithm AST: '+n)
                raise ValueError('protected algorithm source changed: '+n)
        def health(source):
            tree=ast.parse(source);f=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='metrics')
            return ast.dump(f,include_attributes=False)
        name='chapter3_bser/learning/map_contract.py'
        if health(archive.read(name).decode())!=health((root/name).read_text(encoding='utf-8')):raise ValueError('strict health predicates changed')
    expected_new={'chapter3_bser/safety/executor_standby.py','chapter3_bser/learning/standby_experiment.py',
                  'chapter3_bser/learning/executor_standby_source.py'}
    new={n for n in actual.keys()-parent['files'].keys() if n.startswith(('core/','chapter3_bser/'))}
    if new!=expected_new:raise ValueError('unexpected runtime standby extension')
    evo=read(EVOLUTION);indexed={r['path']:r for r in evo['changes']}
    required={n for n in parent['files'].keys()|actual.keys() if parent['files'].get(n)!=actual.get(n)}-{EVOLUTION}
    if len(indexed)!=len(evo['changes']) or set(indexed)!=required or evo['parent_review_sha256']!=parent['sha256']:raise ValueError('standby evolution coverage mismatch')
    for n,row in indexed.items():
        if row.get('before_sha256')!=parent['files'].get(n) or row.get('after_sha256')!=actual.get(n) or not row.get('reason'):raise ValueError('standby evolution hash mismatch: '+n)
    return dict(passed=True,files=len(actual),manifest_sha256=data['sha256'],baseline_archive_verified=True,
        all_parent_evidence_immutable=True,learner_and_physics_byte_identical=True,original_d2_byte_identical=True,
        geometry_kernels_byte_identical=True,strict_health_predicates_preserved=True,
        algorithm_changed=True,change_scope='opt-in search-phase executor navigation guard')
