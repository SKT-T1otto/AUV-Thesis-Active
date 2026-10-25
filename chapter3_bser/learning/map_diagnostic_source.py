"""Current diagnostic-only source identity with immutable route-health parent."""
import hashlib
import json
from pathlib import Path


def verify(root,actual,canonical_digest):
    root=Path(root);read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    parent=read('provenance/map_route_health_manifest.json')
    data=read('provenance/map_diagnostics_manifest.json')
    if data.get('schema')!='auv.active_source.map_diagnostics.v1' or data.get('sha256')!=canonical_digest({k:v for k,v in data.items() if k!='sha256'}):
        raise ValueError('invalid map diagnostics manifest')
    if actual!=data['files']:raise ValueError('active source/config inventory mismatch')
    if data['parent_review_sha256']!=parent['sha256'] or data['parent_manifest_sha256']!=hashlib.sha256((root/'provenance/map_route_health_manifest.json').read_bytes()).hexdigest():
        raise ValueError('immutable route-health parent changed')
    evo=read('provenance/map_diagnostics_evolution.json');indexed={r['path']:r for r in evo['changes']}
    required={n for n in parent['files'].keys()|actual.keys() if parent['files'].get(n)!=actual.get(n)}-{'provenance/map_diagnostics_evolution.json'}
    if len(indexed)!=len(evo['changes']) or set(indexed)!=required or evo['parent_review_sha256']!=parent['sha256']:
        raise ValueError('map diagnostics evolution coverage mismatch')
    for n,row in indexed.items():
        if row.get('before_sha256')!=parent['files'].get(n) or row.get('after_sha256')!=actual.get(n) or not row.get('reason'):
            raise ValueError('map diagnostics evolution hash mismatch: '+n)
    base=read('provenance/baselines/map_route_health_20261025_v1/baseline.json')
    return dict(review_sha256=data['sha256'],profile='active_map_diagnostics_v1',
        inventory_sha256=canonical_digest(actual),parent_commit=base['source_identity']['parent_commit'],distribution=data['schema'])
