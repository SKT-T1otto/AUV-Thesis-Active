"""Exact versioned identity of the opt-in executor standby extension."""
import hashlib
import json
from pathlib import Path


def verify(root,actual,canonical_digest):
    root=Path(root);read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    data=read('provenance/executor_standby_manifest.json');parent=read('provenance/map_diagnostics_manifest.json')
    if data.get('schema')!='auv.active_source.executor_standby.v1' or data.get('sha256')!=canonical_digest({k:v for k,v in data.items() if k!='sha256'}):raise ValueError('invalid standby manifest')
    if actual!=data['files']:raise ValueError('active source/config inventory mismatch')
    if data['parent_review_sha256']!=parent['sha256'] or data['parent_manifest_sha256']!=hashlib.sha256((root/'provenance/map_diagnostics_manifest.json').read_bytes()).hexdigest():raise ValueError('standby parent changed')
    evo=read('provenance/executor_standby_evolution.json');indexed={r['path']:r for r in evo['changes']}
    required={n for n in parent['files'].keys()|actual.keys() if parent['files'].get(n)!=actual.get(n)}-{'provenance/executor_standby_evolution.json'}
    if len(indexed)!=len(evo['changes']) or set(indexed)!=required or evo['parent_review_sha256']!=parent['sha256']:raise ValueError('standby evolution coverage mismatch')
    for n,row in indexed.items():
        if row.get('before_sha256')!=parent['files'].get(n) or row.get('after_sha256')!=actual.get(n) or not row.get('reason'):raise ValueError('standby evolution hash mismatch: '+n)
    return dict(review_sha256=data['sha256'],profile='active_executor_standby_v1',inventory_sha256=canonical_digest(actual),
        parent_commit='72b0167c5c516a43fdd60018ceb955b562525008',distribution=data['schema'])
