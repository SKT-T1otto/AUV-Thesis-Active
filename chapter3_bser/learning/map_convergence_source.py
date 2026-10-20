"""Small-budget experiment successor; previous source and evidence stay immutable."""
import hashlib
import json
from pathlib import Path

MANIFEST='provenance/map_convergence_manifest.json'
EVOLUTION='provenance/map_convergence_evolution.json'
BASELINE='provenance/baselines/map_experiment_20261020_v1'
SCHEMA='auv.active_source.map_convergence.v1'


def verify(root,actual,canonical_digest):
    root=Path(root);read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    data=read(MANIFEST);parent=read('provenance/map_experiment_manifest.json');base=read(BASELINE+'/baseline.json')
    if data.get('schema')!=SCHEMA or data.get('sha256')!=canonical_digest({k:v for k,v in data.items() if k!='sha256'}):raise ValueError('invalid convergence manifest')
    if actual!=data['files']:raise ValueError('active source/config inventory mismatch')
    if (data['parent_review_sha256']!=parent['sha256'] or parent['sha256']!=canonical_digest({k:v for k,v in parent.items() if k!='sha256'})
            or data['parent_manifest_sha256']!=hashlib.sha256((root/'provenance/map_experiment_manifest.json').read_bytes()).hexdigest()
            or base['source_identity']['review_sha256']!=parent['sha256']
            or base['archive_sha256']!=hashlib.sha256((root/BASELINE/'source_and_evidence.zip').read_bytes()).hexdigest()):raise ValueError('immutable experiment parent changed')
    evo=read(EVOLUTION);rows=evo['changes'];indexed={r['path']:r for r in rows}
    expected={p for p in parent['files'].keys()|actual.keys() if parent['files'].get(p)!=actual.get(p)}-{EVOLUTION}
    if len(indexed)!=len(rows) or set(indexed)!=expected or evo['parent_review_sha256']!=parent['sha256']:raise ValueError('convergence evolution coverage mismatch')
    for name,row in indexed.items():
        if row.get('before_sha256')!=parent['files'].get(name) or row.get('after_sha256')!=actual.get(name) or not row.get('reason'):raise ValueError('convergence evolution hash mismatch')
    return dict(review_sha256=data['sha256'],profile='active_map_convergence_v1',inventory_sha256=canonical_digest(actual),
                parent_commit=base['source_identity']['parent_commit'],distribution=SCHEMA)
