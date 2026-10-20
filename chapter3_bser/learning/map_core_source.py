"""Pin the algorithm successor and its immutable Stage-1 parent."""
import hashlib
import json
from pathlib import Path

SCHEMA='auv.active_source.map_core.v1'
MANIFEST='provenance/map_core_manifest.json'
EVOLUTION='provenance/map_core_evolution.json'
BASELINE='provenance/baselines/map_public_interface_20261020_v1'


def verify(root,actual,canonical_digest):
    root=Path(root);read=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
    data=read(MANIFEST);parent=read('provenance/map_stage1_manifest.json')
    if data.get('schema')!=SCHEMA or data.get('sha256')!=canonical_digest({k:v for k,v in data.items() if k!='sha256'}):
        raise ValueError('invalid map core manifest')
    if actual!=data.get('files'):raise ValueError('active source/config inventory mismatch')
    if (parent.get('sha256')!=canonical_digest({k:v for k,v in parent.items() if k!='sha256'})
            or data.get('parent_review_sha256')!=parent['sha256']
            or data.get('parent_manifest_sha256')!=hashlib.sha256((root/'provenance/map_stage1_manifest.json').read_bytes()).hexdigest()):
        raise ValueError('immutable map interface parent changed')
    baseline=read(BASELINE+'/baseline.json')
    if (baseline['source_identity']['review_sha256']!=parent['sha256']
            or baseline['archive_sha256']!=hashlib.sha256((root/BASELINE/'source_and_evidence.zip').read_bytes()).hexdigest()):
        raise ValueError('immutable map interface archive changed')
    evo=read(EVOLUTION);rows=evo['changes'];indexed={r['path']:r for r in rows}
    required={p for p in parent['files'].keys()|actual.keys() if parent['files'].get(p)!=actual.get(p)}-{EVOLUTION}
    if (evo.get('algorithm_changed') is not True or evo.get('parent_review_sha256')!=parent['sha256']
            or len(rows)!=len(indexed) or set(indexed)!=required):raise ValueError('core evolution coverage mismatch')
    for name,row in indexed.items():
        if row.get('before_sha256')!=parent['files'].get(name) or row.get('after_sha256')!=actual.get(name) or not row.get('reason'):
            raise ValueError('core evolution hash/reason mismatch: '+name)
    return dict(review_sha256=data['sha256'],profile='active_map_core_v1',inventory_sha256=canonical_digest(actual),
                parent_commit=baseline['source_identity']['parent_commit'],distribution=SCHEMA)
