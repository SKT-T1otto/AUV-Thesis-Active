"""Independent OLD executor successor; parent source is data, never imported."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile
from tools.executor_standby_source import inventory as parent_inventory
from tools.map_interface_source import digest,legacy_ast

MANIFEST='provenance/executor_recovery_manifest.json'
EVOLUTION='provenance/executor_recovery_evolution.json'
PARENT='provenance/executor_standby_manifest.json'
BASELINE='provenance/baselines/executor_standby_20261027_v1'


def inventory(root):
    root=Path(root);actual=parent_inventory(root)
    for name in (PARENT,EVOLUTION):actual[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    return actual


def verify(root):
    root=Path(root);read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    sha=lambda n:hashlib.sha256((root/n).read_bytes()).hexdigest()
    actual=inventory(root);data,parent,base=read(MANIFEST),read(PARENT),read(BASELINE+'/baseline.json')
    if data.get('schema')!='auv.active_source.executor_recovery.v2' or data.get('sha256')!=digest({k:v for k,v in data.items() if k!='sha256'}):raise ValueError('invalid executor recovery manifest')
    if actual!=data['files']:raise ValueError('independent source inventory mismatch')
    if (data['parent_review_sha256']!=parent['sha256'] or data['parent_manifest_sha256']!=sha(PARENT)
        or base['source_identity']['review_sha256']!=parent['sha256'] or sha(BASELINE+'/source_and_docs.zip')!=base['archive_sha256']):
        raise ValueError('immutable standby parent changed')
    allowed={'chapter3_bser/planner_factory.py','chapter3_bser/learning/map_contract.py','chapter3_bser/learning/map_suite.py',
        'chapter3_bser/learning/runtime.py','chapter3_bser/learning/provenance.py'}
    with zipfile.ZipFile(root/BASELINE/'source_and_docs.zip') as archive:
        external=base['external_immutable_files']
        if set(archive.namelist())!=set(base['files'])-set(external):raise ValueError('parent archive inventory changed')
        for name,h in base['files'].items():
            saved=sha(name) if name in external else hashlib.sha256(archive.read(name)).hexdigest()
            if name in external and saved!=h:raise ValueError('parent evidence changed: '+name)
            if saved!=h:raise ValueError('frozen parent member changed: '+name)
        for name in ('core/env/uav_env.py','core/env/mission_env.py','chapter3_bser/learning/base_runtime.py'):
            if legacy_ast(archive.read(name).decode())!=legacy_ast((root/name).read_text(encoding='utf-8')):raise ValueError('legacy AST changed: '+name)
        for name,h in parent['files'].items():
            if name.startswith(('core/','chapter3_bser/')) and name not in allowed and actual.get(name)!=h:
                if 'geometry_hypotheses.py' in name:raise ValueError('protected geometry kernel changed: '+name)
                if 'bounded_belief.py' in name:raise ValueError('observer changed algorithm AST: '+name)
                raise ValueError('protected algorithm source changed: '+name)
        def function(source,name):
            return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        name='chapter3_bser/learning/map_contract.py'
        if function(archive.read(name).decode(),'metrics')!=function((root/name).read_text(encoding='utf-8'),'metrics'):
            raise ValueError('strict health predicates changed')
        name='chapter3_bser/learning/runtime.py'
        new=ast.parse((root/name).read_text(encoding='utf-8'));old=ast.parse(archive.read(name).decode())
        for cls in new.body:
            if isinstance(cls,ast.ClassDef):cls.body=[n for n in cls.body if not isinstance(n,ast.FunctionDef) or n.name!='_gates']
        if ast.dump(old,include_attributes=False)!=ast.dump(new,include_attributes=False):raise ValueError('runtime changed beyond explicit phase gate')
    expected={'chapter3_bser/safety/executor_public.py','chapter3_bser/safety/executor_recovery.py',
        'chapter3_bser/safety/executor_recovery_options.py','chapter3_bser/learning/executor_experiment.py',
        'chapter3_bser/learning/executor_recovery_source.py'}
    if {n for n in actual.keys()-parent['files'].keys() if n.startswith(('core/','chapter3_bser/'))}!=expected:
        raise ValueError('unexpected runtime recovery extension')
    evo=read(EVOLUTION);indexed={r['path']:r for r in evo['changes']}
    required={n for n in parent['files'].keys()|actual.keys() if parent['files'].get(n)!=actual.get(n)}-{EVOLUTION}
    if len(indexed)!=len(evo['changes']) or set(indexed)!=required or evo['parent_review_sha256']!=parent['sha256']:raise ValueError('recovery evolution coverage mismatch')
    for name,row in indexed.items():
        if row.get('before_sha256')!=parent['files'].get(name) or row.get('after_sha256')!=actual.get(name) or not row.get('reason'):raise ValueError('recovery evolution hash mismatch: '+name)
    return dict(passed=True,files=len(actual),manifest_sha256=data['sha256'],all_parent_evidence_immutable=True,
        baseline_archive_verified=True,original_d2_byte_identical=True,physics_and_learner_optimizers_byte_identical=True,
        original_v1_guard_byte_identical=True,geometry_kernels_byte_identical=True,strict_health_predicates_preserved=True,
        algorithm_changed=True,scope='OLD-only executor navigation and explicit phase-specific runtime residual gate')
