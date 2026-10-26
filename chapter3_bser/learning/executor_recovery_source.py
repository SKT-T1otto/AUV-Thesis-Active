"""Exact runtime identity/evolution of OLD-only executor recovery v2."""
import hashlib
import json
from pathlib import Path


def verify(root,actual,canonical_digest):
    root=Path(root);read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    data=read('provenance/executor_recovery_manifest.json');parent=read('provenance/executor_standby_manifest.json')
    if data.get('schema')!='auv.active_source.executor_recovery.v2' or data.get('sha256')!=canonical_digest({k:v for k,v in data.items() if k!='sha256'}):raise ValueError('invalid recovery manifest')
    if actual!=data['files']:raise ValueError('active source/config inventory mismatch')
    if data['parent_review_sha256']!=parent['sha256'] or data['parent_manifest_sha256']!=hashlib.sha256((root/'provenance/executor_standby_manifest.json').read_bytes()).hexdigest():raise ValueError('recovery parent changed')
    evo=read('provenance/executor_recovery_evolution.json');indexed={r['path']:r for r in evo['changes']}
    required={n for n in parent['files'].keys()|actual.keys() if parent['files'].get(n)!=actual.get(n)}-{'provenance/executor_recovery_evolution.json'}
    if len(indexed)!=len(evo['changes']) or set(indexed)!=required or evo['parent_review_sha256']!=parent['sha256']:raise ValueError('recovery evolution coverage mismatch')
    for name,row in indexed.items():
        if row.get('before_sha256')!=parent['files'].get(name) or row.get('after_sha256')!=actual.get(name) or not row.get('reason'):raise ValueError('recovery evolution hash mismatch: '+name)
    return dict(review_sha256=data['sha256'],profile='active_executor_recovery_v2',inventory_sha256=canonical_digest(actual),
        parent_commit='aef6833a6b80dcede92561b6497aafc1d5ad60e7',distribution=data['schema'])
