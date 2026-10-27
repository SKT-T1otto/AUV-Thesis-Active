"""Verification-only successor identity; safety algorithm remains byte-identical."""
import hashlib,json
from pathlib import Path

def verify(root,actual,canonical_digest):
    root=Path(root);read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    data=read('provenance/safe_rl_transport_manifest.json');parent=read('provenance/safe_rl_recovery_manifest.json')
    if data.get('schema')!='auv.active_source.safe_rl_transport.v1' or data.get('sha256')!=canonical_digest({k:v for k,v in data.items() if k!='sha256'}):raise ValueError('invalid safety validation manifest')
    if actual!=data['files']:raise ValueError('active source/config inventory mismatch')
    if data.get('algorithm_changed') is not False or data['parent_review_sha256']!=parent['sha256'] or data['parent_manifest_sha256']!=hashlib.sha256((root/'provenance/safe_rl_recovery_manifest.json').read_bytes()).hexdigest():raise ValueError('immutable safety-RL parent changed')
    evo=read('provenance/safe_rl_transport_evolution.json');indexed={r['path']:r for r in evo['changes']}
    required={n for n in parent['files'].keys()|actual.keys() if parent['files'].get(n)!=actual.get(n)}-{'provenance/safe_rl_transport_evolution.json'}
    if len(indexed)!=len(evo['changes']) or set(indexed)!=required or evo['parent_review_sha256']!=parent['sha256']:raise ValueError('safety validation evolution coverage mismatch')
    for name,row in indexed.items():
        if row.get('before_sha256')!=parent['files'].get(name) or row.get('after_sha256')!=actual.get(name) or not row.get('reason'):raise ValueError('safety validation evolution hash mismatch: '+name)
    return dict(review_sha256=data['sha256'],profile='active_safe_rl_transport_v1',inventory_sha256=canonical_digest(actual),
        parent_commit='b57d8e5c9f776c912138a71dc613e3bfa4f29fa4',distribution=data['schema'])
