"""Exact identity for public local geometry; no old policy relabeling."""
import hashlib
import json
from pathlib import Path


def verify(root, actual, canonical_digest):
    root = Path(root)
    read = lambda n: json.loads((root/n).read_text(encoding='utf-8'))
    data = read('provenance/reach_avoid_geometry_manifest.json')
    parent = read('provenance/reach_avoid_manifest.json')
    if data.get('schema') != 'auv.active_source.reach_avoid_geometry.v2' or data.get('sha256') != canonical_digest({k:v for k,v in data.items() if k!='sha256'}):
        raise ValueError('invalid scan geometry manifest')
    if actual != data['files']:
        raise ValueError('active source/config inventory mismatch')
    if data.get('algorithm_changed') is not True or data['parent_review_sha256']!=parent['sha256'] or data['parent_manifest_sha256']!=hashlib.sha256((root/'provenance/reach_avoid_manifest.json').read_bytes()).hexdigest():
        raise ValueError('immutable reach-avoid parent changed')
    evo = read('provenance/reach_avoid_geometry_evolution.json')
    indexed = {r['path']:r for r in evo['changes']}
    required = {n for n in parent['files'].keys()|actual.keys() if parent['files'].get(n)!=actual.get(n)}-{'provenance/reach_avoid_geometry_evolution.json'}
    if len(indexed)!=len(evo['changes']) or set(indexed)!=required or evo['parent_review_sha256']!=parent['sha256']:
        raise ValueError('geometry evolution coverage mismatch')
    for name,row in indexed.items():
        if row.get('before_sha256')!=parent['files'].get(name) or row.get('after_sha256')!=actual.get(name) or not row.get('reason'):
            raise ValueError('geometry evolution hash mismatch: '+name)
    return dict(review_sha256=data['sha256'], profile='active_reach_avoid_geometry_v2',
        inventory_sha256=canonical_digest(actual), parent_commit='806635eceedec881d33983b1ff8a794abd29fc09', distribution=data['schema'])
