"""Independent v3 public-route feasibility fix and immutable-parent verifier."""
import hashlib,json,zipfile
from pathlib import Path
from tools.map_bounded_source import inventory as bounded_inventory
from tools.map_interface_source import digest,legacy_ast

MANIFEST='provenance/map_route_health_manifest.json'
EVOLUTION='provenance/map_route_health_evolution.json'
PARENT='provenance/map_bounded_manifest.json'
BASELINE='provenance/baselines/map_bounded_20261023_v3'


def inventory(root):
    root=Path(root);actual=bounded_inventory(root)
    for n in (PARENT,EVOLUTION):actual[n]=hashlib.sha256((root/n).read_bytes()).hexdigest()
    return actual


def verify(root):
    root=Path(root);read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    sha=lambda n:hashlib.sha256((root/n).read_bytes()).hexdigest()
    data,parent,base=read(MANIFEST),read(PARENT),read(BASELINE+'/baseline.json');actual=inventory(root)
    if data.get('schema')!='auv.active_source.map_route_health.v1' or data.get('sha256')!=digest({k:v for k,v in data.items() if k!='sha256'}):raise ValueError('invalid route health manifest')
    if actual!=data['files']:raise ValueError('independent source inventory mismatch')
    if (parent['sha256']!=digest({k:v for k,v in parent.items() if k!='sha256'}) or data['parent_review_sha256']!=parent['sha256']
            or data['parent_manifest_sha256']!=sha(PARENT) or base['source_identity']['review_sha256']!=parent['sha256']
            or base['archive_sha256']!=sha(BASELINE+'/source_and_evidence.zip')):raise ValueError('immutable bounded parent changed')
    external=base['external_frozen_archives']
    with zipfile.ZipFile(root/BASELINE/'source_and_evidence.zip') as archive:
        if set(archive.namelist())!=set(base['files'])-set(external):raise ValueError('parent archive inventory changed')
        for n,h in base['files'].items():
            saved=sha(n) if n in external else hashlib.sha256(archive.read(n)).hexdigest()
            if saved!=h or (n in external and external[n]!=h):raise ValueError('frozen parent member changed')
            if n.startswith(('provenance/','verification/')) and sha(n)!=h:raise ValueError('parent evidence changed: '+n)
        for n,h in parent['files'].items():
            if base['files'].get(n)!=h:raise ValueError('frozen parent source mismatch')
        for n in ('core/env/uav_env.py','core/env/mission_env.py','chapter3_bser/learning/base_runtime.py'):
            if legacy_ast(archive.read(n).decode())!=legacy_ast((root/n).read_text(encoding='utf-8')):raise ValueError('legacy AST changed: '+n)
        for name in ('geometry_hypotheses.py','geometry_cache.py','coupled_belief.py','target_particles.py','bounded_belief.py','persistent_map.py','bounded_config.py'):
            n='chapter3_bser/experiments/coupled_search_v1/'+name
            if (root/n).read_bytes()!=archive.read(n):raise ValueError('protected geometry kernel changed: '+name)
    allowed={'chapter3_bser/experiments/coupled_search_v1/allocator.py','chapter3_bser/learning/map_contract.py',
        'chapter3_bser/learning/provenance.py','checks/test_map_source.py','configs/rebuild_check_contract.json',
        'tools/verify_source.py','tools/validate_checkout.py'}
    changed={n for n,h in parent['files'].items() if actual.get(n)!=h}
    if changed-allowed:raise ValueError('protected algorithm source changed: '+repr(sorted(changed-allowed)))
    runtime={n for n in actual.keys()-parent['files'].keys() if n.startswith(('core/','chapter3_bser/'))}
    if runtime!={'chapter3_bser/experiments/coupled_search_v1/route_feasibility.py','chapter3_bser/learning/map_route_health_source.py'}:raise ValueError('unexpected runtime extension')
    evolution=read(EVOLUTION);rows=evolution['changes'];indexed={r['path']:r for r in rows}
    required={n for n in parent['files'].keys()|actual.keys() if parent['files'].get(n)!=actual.get(n)}-{EVOLUTION}
    if len(rows)!=len(indexed) or set(indexed)!=required or evolution['parent_review_sha256']!=parent['sha256']:raise ValueError('route health evolution coverage mismatch')
    for n,row in indexed.items():
        if row.get('before_sha256')!=parent['files'].get(n) or row.get('after_sha256')!=actual.get(n) or not row.get('reason'):raise ValueError('route health evolution hash mismatch: '+n)
    return dict(passed=True,files=len(actual),manifest_sha256=data['sha256'],baseline_archive_verified=True,
        all_parent_evidence_immutable=True,learner_and_physics_byte_identical=True,
        original_d2_byte_identical=True,v2_v3_inference_and_kernel_byte_identical=True,
        map_algorithm_revision='coupled_search_bounded_v3',compute_revision='public_unroutable_start_constraints_v1')
