"""Reviewed continuous-search successor; never relabel an older checkpoint."""
import hashlib
import json
from pathlib import Path

MANIFEST='provenance/map_persistence_manifest.json'
EVOLUTION='provenance/map_persistence_evolution.json'
BASELINE='provenance/baselines/map_convergence_20261021_v1'
SCHEMA='auv.active_source.map_persistence.v1'


def verify(root,actual,canonical_digest):
    root=Path(root);read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    data=read(MANIFEST);parent=read('provenance/map_convergence_manifest.json');base=read(BASELINE+'/baseline.json')
    if data.get('schema')!=SCHEMA or data.get('sha256')!=canonical_digest({k:v for k,v in data.items() if k!='sha256'}):raise ValueError('invalid persistence manifest')
    if actual!=data['files']:raise ValueError('active source/config inventory mismatch')
    if (data['parent_review_sha256']!=parent['sha256'] or base['source_identity']['review_sha256']!=parent['sha256']
            or data['parent_manifest_sha256']!=hashlib.sha256((root/'provenance/map_convergence_manifest.json').read_bytes()).hexdigest()
            or base['archive_sha256']!=hashlib.sha256((root/BASELINE/'source_and_evidence.zip').read_bytes()).hexdigest()):raise ValueError('immutable convergence parent changed')
    evo=read(EVOLUTION);rows=evo['changes'];indexed={r['path']:r for r in rows}
    required={p for p in parent['files'].keys()|actual.keys() if parent['files'].get(p)!=actual.get(p)}-{EVOLUTION}
    if len(rows)!=len(indexed) or set(indexed)!=required or evo['parent_review_sha256']!=parent['sha256']:raise ValueError('persistence evolution coverage mismatch')
    for n,row in indexed.items():
        if row.get('before_sha256')!=parent['files'].get(n) or row.get('after_sha256')!=actual.get(n) or not row.get('reason'):raise ValueError('persistence evolution hash mismatch')
    return dict(review_sha256=data['sha256'],profile='active_map_persistence_v1',inventory_sha256=canonical_digest(actual),
                parent_commit=base['source_identity']['parent_commit'],distribution=SCHEMA)
