"""Independent filesystem/AST verification for native map training scripts."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile
from tools.map_interface_source import legacy_ast,digest

MANIFEST='provenance/map_experiment_manifest.json'
EVOLUTION='provenance/map_experiment_evolution.json'
BASELINE='provenance/baselines/map_core_20261020_v1'


def method_ast(source,name):
    tree=ast.parse(source)
    return ast.dump(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)


def verify(root):
    root=Path(root);read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    data=read(MANIFEST);parent_bytes=(root/'provenance/map_core_manifest.json').read_bytes();parent=json.loads(parent_bytes)
    if data.get('schema')!='auv.active_source.map_experiment.v1' or data.get('sha256')!=digest({k:v for k,v in data.items() if k!='sha256'}):raise ValueError('invalid independent map experiment identity')
    actual={}
    for folder in ('core','chapter3_bser','configs','checks','tools','provenance/parent','provenance/history','provenance/baselines'):
        for file in (root/folder).rglob('*'):
            if file.is_file() and file.suffix in ('.py','.json','.zip'):actual[file.relative_to(root).as_posix()]=hashlib.sha256(file.read_bytes()).hexdigest()
    for file in (root/'scripts').rglob('*.sh'):actual[file.relative_to(root).as_posix()]=hashlib.sha256(file.read_bytes()).hexdigest()
    for name in ('cli.py','project_paths.py','requirements.txt','AGENTS.md','.gitattributes','.gitignore',
                 'provenance/rebuild_evolution.json','provenance/map_stage1_evolution.json','provenance/map_stage1_manifest.json',
                 'provenance/map_core_evolution.json','provenance/map_core_manifest.json',EVOLUTION):actual[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual!=data['files']:raise ValueError('independent source inventory mismatch')
    if data['parent_manifest_sha256']!=hashlib.sha256(parent_bytes).hexdigest() or data['parent_review_sha256']!=parent['sha256'] or parent['sha256']!=digest({k:v for k,v in parent.items() if k!='sha256'}):raise ValueError('immutable map core manifest changed')
    base=read(BASELINE+'/baseline.json');archive=root/BASELINE/'source_and_evidence.zip'
    if base['source_identity']['review_sha256']!=parent['sha256'] or base['archive_sha256']!=hashlib.sha256(archive.read_bytes()).hexdigest():raise ValueError('map core baseline archive changed')
    proofs=[]
    with zipfile.ZipFile(archive) as frozen:
        if set(frozen.namelist())!=set(base['files']):raise ValueError('map core archive inventory changed')
        for name,expected in base['files'].items():
            if hashlib.sha256(frozen.read(name)).hexdigest()!=expected:raise ValueError('map core frozen member changed')
            if name.startswith(('verification/','provenance/')) and hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:raise ValueError('immutable parent evidence changed: '+name)
        for name,expected in parent['files'].items():
            if hashlib.sha256(frozen.read(name)).hexdigest()!=expected:raise ValueError('frozen map core source differs')
        for name in ('core/env/uav_env.py','core/env/mission_env.py','chapter3_bser/learning/base_runtime.py'):
            if legacy_ast(frozen.read(name).decode())!=legacy_ast((root/name).read_text(encoding='utf-8')):raise ValueError('legacy AST changed: '+name)
            proofs.append(name)
        name='chapter3_bser/learning/train.py'
        for method in ('_update','_perform_update','_run_loop'):
            if method_ast(frozen.read(name).decode(),method)!=method_ast((root/name).read_text(encoding='utf-8'),method):raise ValueError('learning update/schedule AST changed: '+method)
            proofs.append(name+':'+method)
    permitted={'cli.py','chapter3_bser/learning/config.py','chapter3_bser/learning/provenance.py','chapter3_bser/learning/train.py',
        'chapter3_bser/learning/evaluate.py','chapter3_bser/learning/reference.py','chapter3_bser/learning/evaluation_records.py',
        'tools/verify_source.py','tools/run_checks.py','tools/validate_checkout.py','checks/test_map_source.py','configs/rebuild_check_contract.json'}
    changed={p for p in parent['files'] if parent['files'][p]!=actual.get(p)}
    if changed-permitted:raise ValueError('protected algorithm source changed: '+repr(sorted(changed-permitted)))
    if any(p.startswith('core/') and p not in parent['files'] for p in actual):raise ValueError('unexpected physical core extension')
    evo=read(EVOLUTION);rows=evo['changes'];indexed={r['path']:r for r in rows}
    required={p for p in parent['files'].keys()|actual.keys() if parent['files'].get(p)!=actual.get(p)}-{EVOLUTION}
    if len(indexed)!=len(rows) or set(indexed)!=required or evo.get('parent_review_sha256')!=parent['sha256']:raise ValueError('independent map experiment evolution mismatch')
    for name,row in indexed.items():
        if row.get('before_sha256')!=parent['files'].get(name) or row.get('after_sha256')!=actual.get(name) or not row.get('reason'):raise ValueError('independent map experiment hash/reason mismatch')
    return dict(passed=True,files=len(actual),manifest_sha256=data['sha256'],baseline_archive_verified=True,legacy_ast_equal=proofs,
                map_core_unchanged=True,learning_updates_unchanged=True)
