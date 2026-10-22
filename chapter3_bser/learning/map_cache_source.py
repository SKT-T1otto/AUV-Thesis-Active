"""Exact identity for cache-only evolution; earlier manifests stay immutable."""
import hashlib
import json
from pathlib import Path

MANIFEST = 'provenance/map_cache_manifest.json'
EVOLUTION = 'provenance/map_cache_evolution.json'
PARENT = 'provenance/map_persistence_manifest.json'
BASELINE = 'provenance/baselines/map_persistence_20261022_v1'
SCHEMA = 'auv.active_source.map_cache.v1'


def verify(root, actual, canonical_digest):
    root = Path(root)
    read = lambda name: json.loads((root/name).read_text(encoding='utf-8'))
    sha = lambda name: hashlib.sha256((root/name).read_bytes()).hexdigest()
    data, parent, base = read(MANIFEST), read(PARENT), read(BASELINE+'/baseline.json')
    if data.get('schema') != SCHEMA or data.get('sha256') != canonical_digest({k:v for k,v in data.items() if k != 'sha256'}):
        raise ValueError('invalid cache evolution manifest')
    if actual != data['files']:
        raise ValueError('active source/config inventory mismatch')
    if (data['parent_review_sha256'] != parent['sha256'] or data['parent_manifest_sha256'] != sha(PARENT)
            or base['source_identity']['review_sha256'] != parent['sha256']
            or base['archive_sha256'] != sha(BASELINE+'/source_and_evidence.zip')):
        raise ValueError('immutable persistence parent changed')
    evolution = read(EVOLUTION)
    rows = evolution['changes']; indexed = {row['path']:row for row in rows}
    required = {name for name in parent['files'].keys() | actual.keys()
                if parent['files'].get(name) != actual.get(name)} - {EVOLUTION}
    if len(rows) != len(indexed) or set(indexed) != required or evolution['parent_review_sha256'] != parent['sha256']:
        raise ValueError('cache evolution coverage mismatch')
    for name, row in indexed.items():
        if row.get('before_sha256') != parent['files'].get(name) or row.get('after_sha256') != actual.get(name) or not row.get('reason'):
            raise ValueError('cache evolution hash mismatch: '+name)
    return dict(review_sha256=data['sha256'], profile='active_map_cache_v1',
                inventory_sha256=canonical_digest(actual), parent_commit=base['source_identity']['parent_commit'],
                distribution=SCHEMA)
