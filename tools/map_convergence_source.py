"""Independent byte/AST proof: only budgets and offline diagnostics evolve."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile
from tools.map_interface_source import digest,legacy_ast

MANIFEST='provenance/map_convergence_manifest.json'
EVOLUTION='provenance/map_convergence_evolution.json'
BASELINE='provenance/baselines/map_experiment_20261020_v1'


def verify(root):
    root=Path(root);read=lambda n:json.loads((root/n).read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    data=read(MANIFEST);parent=read('provenance/map_experiment_manifest.json');base=read(BASELINE+'/baseline.json')
    actual={}
    for folder in ('core','chapter3_bser','configs','checks','tools','provenance/parent','provenance/history','provenance/baselines'):
        for p in (root/folder).rglob('*'):
            if p.is_file() and p.suffix in ('.py','.json','.zip'):actual[p.relative_to(root).as_posix()]=sha(p)
    for p in (root/'scripts').rglob('*.sh'):actual[p.relative_to(root).as_posix()]=sha(p)
    for n in ('cli.py','project_paths.py','requirements.txt','AGENTS.md','.gitattributes','.gitignore',
        'provenance/rebuild_evolution.json','provenance/map_stage1_evolution.json','provenance/map_stage1_manifest.json',
        'provenance/map_core_evolution.json','provenance/map_core_manifest.json','provenance/map_experiment_evolution.json',
        'provenance/map_experiment_manifest.json',EVOLUTION):actual[n]=sha(root/n)
    if data.get('schema')!='auv.active_source.map_convergence.v1' or data.get('sha256')!=digest({k:v for k,v in data.items() if k!='sha256'}):raise ValueError('invalid independent convergence manifest')
    if actual!=data['files']:raise ValueError('independent source inventory mismatch')
    if (data['parent_review_sha256']!=parent['sha256'] or data['parent_manifest_sha256']!=sha(root/'provenance/map_experiment_manifest.json')
            or parent['sha256']!=digest({k:v for k,v in parent.items() if k!='sha256'}) or base['source_identity']['review_sha256']!=parent['sha256']
            or sha(root/BASELINE/'source_and_evidence.zip')!=base['archive_sha256']):raise ValueError('immutable experiment parent changed')
    with zipfile.ZipFile(root/BASELINE/'source_and_evidence.zip') as archive:
        if set(archive.namelist())!=set(base['files']):raise ValueError('parent archive inventory changed')
        for n,h in base['files'].items():
            if hashlib.sha256(archive.read(n)).hexdigest()!=h:raise ValueError('frozen parent member changed')
            if n.startswith(('provenance/','verification/')) and sha(root/n)!=h:raise ValueError('immutable parent evidence changed: '+n)
        for n,h in parent['files'].items():
            if hashlib.sha256(archive.read(n)).hexdigest()!=h:raise ValueError('frozen parent source changed')
        for n in ('core/env/uav_env.py','core/env/mission_env.py','chapter3_bser/learning/base_runtime.py'):
            if legacy_ast(archive.read(n).decode())!=legacy_ast((root/n).read_text(encoding='utf-8')):raise ValueError('legacy AST changed: '+n)
    permitted={'chapter3_bser/learning/provenance.py','chapter3_bser/learning/map_suite.py','chapter3_bser/learning/map_report.py',
        'tools/verify_source.py','tools/validate_checkout.py','tools/check_map_training.py','checks/test_map_source.py',
        'checks/test_map_suite.py','configs/map_comparison_v1.json','configs/map_comparison_pilot_v1.json','configs/rebuild_check_contract.json','scripts/linux_map_experiment.sh'}
    changed={n for n,h in parent['files'].items() if actual.get(n)!=h}
    if changed-permitted:raise ValueError('protected algorithm source changed: '+repr(sorted(changed-permitted)))
    new_runtime={n for n in actual.keys()-parent['files'].keys() if n.startswith(('core/','chapter3_bser/'))}
    if new_runtime-{'chapter3_bser/learning/map_convergence.py','chapter3_bser/learning/map_convergence_source.py'}:raise ValueError('unexpected runtime extension')
    evo=read(EVOLUTION);rows=evo['changes'];indexed={r['path']:r for r in rows}
    required={n for n in parent['files'].keys()|actual.keys() if parent['files'].get(n)!=actual.get(n)}-{EVOLUTION}
    if set(indexed)!=required or len(indexed)!=len(rows) or evo['parent_review_sha256']!=parent['sha256']:raise ValueError('independent convergence evolution coverage mismatch')
    for n,row in indexed.items():
        if row.get('before_sha256')!=parent['files'].get(n) or row.get('after_sha256')!=actual.get(n) or not row.get('reason'):raise ValueError('independent convergence evolution hash mismatch')
    return dict(passed=True,files=len(actual),manifest_sha256=data['sha256'],baseline_archive_verified=True,
                map_core_unchanged=True,learner_and_update_schedule_byte_identical=True,all_parent_evidence_immutable=True)
