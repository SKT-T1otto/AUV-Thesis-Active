"""Runtime identity for the public-route constraint correction."""
import hashlib,json
from pathlib import Path


def verify(root,actual,canonical_digest):
    root=Path(root);read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    parent=read('provenance/map_bounded_manifest.json');data=read('provenance/map_route_health_manifest.json')
    base=read('provenance/baselines/map_bounded_20261023_v3/baseline.json')
    if data.get('schema')!='auv.active_source.map_route_health.v1' or data.get('sha256')!=canonical_digest({k:v for k,v in data.items() if k!='sha256'}):raise ValueError('invalid route health source manifest')
    if actual!=data['files']:raise ValueError('active source/config inventory mismatch')
    if (data['parent_review_sha256']!=parent['sha256'] or base['source_identity']['review_sha256']!=parent['sha256']
            or data['parent_manifest_sha256']!=hashlib.sha256((root/'provenance/map_bounded_manifest.json').read_bytes()).hexdigest()):raise ValueError('immutable bounded parent changed')
    evo=read('provenance/map_route_health_evolution.json');rows=evo['changes'];indexed={r['path']:r for r in rows}
    required={n for n in parent['files'].keys()|actual.keys() if parent['files'].get(n)!=actual.get(n)}-{'provenance/map_route_health_evolution.json'}
    if len(rows)!=len(indexed) or set(indexed)!=required or evo['parent_review_sha256']!=parent['sha256']:raise ValueError('route health evolution coverage mismatch')
    for n,row in indexed.items():
        if row.get('before_sha256')!=parent['files'].get(n) or row.get('after_sha256')!=actual.get(n) or not row.get('reason'):raise ValueError('route health evolution hash mismatch: '+n)
    return dict(review_sha256=data['sha256'],profile='active_map_route_health_v1',inventory_sha256=canonical_digest(actual),
        parent_commit=base['source_identity']['parent_commit'],distribution=data['schema'])
