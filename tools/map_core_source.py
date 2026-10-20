"""Independent successor verifier: filesystem/AST only, no runtime imports."""
import hashlib
import json
from pathlib import Path
import zipfile
from tools.map_interface_source import legacy_ast, digest

BASELINE='provenance/baselines/map_public_interface_20261020_v1'
EVOLUTION='provenance/map_core_evolution.json'
MANIFEST='provenance/map_core_manifest.json'


def verify(root):
    root=Path(root);read=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
    data=read(MANIFEST)
    if data.get('schema')!='auv.active_source.map_core.v1' or data.get('sha256')!=digest({k:v for k,v in data.items() if k!='sha256'}):
        raise ValueError('invalid independent map core identity')
    actual={}
    for folder in ('core','chapter3_bser','configs','checks','tools','provenance/parent','provenance/history','provenance/baselines'):
        for file in (root/folder).rglob('*'):
            if file.is_file() and file.suffix in ('.py','.json','.zip'):
                actual[file.relative_to(root).as_posix()]=hashlib.sha256(file.read_bytes()).hexdigest()
    for name in ('cli.py','project_paths.py','requirements.txt','AGENTS.md','.gitattributes','.gitignore',
                 'provenance/rebuild_evolution.json','provenance/map_stage1_evolution.json','provenance/map_stage1_manifest.json',EVOLUTION):
        actual[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual!=data['files']:raise ValueError('independent source inventory mismatch')
    parent_bytes=(root/'provenance/map_stage1_manifest.json').read_bytes();parent=json.loads(parent_bytes)
    if (hashlib.sha256(parent_bytes).hexdigest()!=data.get('parent_manifest_sha256')
            or parent['sha256']!=data.get('parent_review_sha256')
            or parent['sha256']!=digest({k:v for k,v in parent.items() if k!='sha256'})):
        raise ValueError('immutable interface parent changed')
    baseline=read(BASELINE+'/baseline.json');archive=root/BASELINE/'source_and_evidence.zip'
    if (baseline['archive_sha256']!=hashlib.sha256(archive.read_bytes()).hexdigest()
            or baseline['source_identity']['review_sha256']!=parent['sha256']):raise ValueError('parent archive changed')
    with zipfile.ZipFile(archive) as frozen:
        if set(frozen.namelist())!=set(baseline['files']):raise ValueError('parent archive inventory changed')
        for name,sha in baseline['files'].items():
            if hashlib.sha256(frozen.read(name)).hexdigest()!=sha:raise ValueError('parent archive member changed')
            if (name.startswith(('verification/','provenance/')) and name not in (EVOLUTION,)
                    and hashlib.sha256((root/name).read_bytes()).hexdigest()!=sha):
                raise ValueError('immutable parent evidence changed: '+name)
        if frozen.read('provenance/map_stage1_manifest.json')!=parent_bytes:raise ValueError('frozen manifest differs')
        for name,sha in parent['files'].items():
            if hashlib.sha256(frozen.read(name)).hexdigest()!=sha:raise ValueError('frozen source differs from parent')
        proofs=[]
        for name in ('core/env/uav_env.py','core/env/mission_env.py','chapter3_bser/learning/base_runtime.py'):
            if legacy_ast(frozen.read(name).decode('utf-8'))!=legacy_ast((root/name).read_text(encoding='utf-8')):
                raise ValueError('legacy AST changed outside public hooks: '+name)
            proofs.append(name)
    permitted={'core/env/mission_env.py','chapter3_bser/planner_factory.py','chapter3_bser/learning/provenance.py',
               'tools/verify_source.py','configs/rebuild_check_contract.json','checks/test_map_source.py'}
    changed={p for p in parent['files'] if parent['files'][p]!=actual.get(p)}
    unexpected={p for p in changed if p not in permitted and not p.startswith('chapter3_bser/experiments/coupled_search_v1/')}
    if unexpected:raise ValueError('protected legacy source changed: '+repr(sorted(unexpected)))
    if any(p.startswith('core/') and p not in parent['files'] for p in actual):raise ValueError('unexpected core extension')
    evo=read(EVOLUTION);rows=evo['changes'];indexed={r['path']:r for r in rows}
    required={p for p in parent['files'].keys()|actual.keys() if parent['files'].get(p)!=actual.get(p)}-{EVOLUTION}
    if (len(rows)!=len(indexed) or set(indexed)!=required or evo.get('algorithm_changed') is not True
            or evo.get('parent_review_sha256')!=parent['sha256']):raise ValueError('independent core evolution mismatch')
    for name,row in indexed.items():
        if row.get('before_sha256')!=parent['files'].get(name) or row.get('after_sha256')!=actual.get(name) or not row.get('reason'):
            raise ValueError('independent core evolution hash mismatch: '+name)
    return dict(passed=True,files=len(actual),manifest_sha256=data['sha256'],baseline_archive_verified=True,
                legacy_ast_equal=proofs,algorithm_changed=True,default_enabled=False)
