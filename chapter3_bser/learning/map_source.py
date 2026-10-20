"""Validate the observation-interface successor without rewriting its parent."""
import hashlib
import json
from pathlib import Path

SCHEMA = 'auv.active_source.map_interface.v1'
MANIFEST = 'provenance/map_stage1_manifest.json'
EVOLUTION = 'provenance/map_stage1_evolution.json'
BASELINE = 'provenance/baselines/map_d2_repair_v2_20261020_v1'


def verify(root, actual, canonical_digest):
    root = Path(root).resolve()
    data = json.loads((root/MANIFEST).read_text(encoding='utf-8'))
    parent_path = root/'provenance/active_manifest.json'
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    body = {k: v for k, v in data.items() if k != 'sha256'}
    if data.get('schema') != SCHEMA or data.get('sha256') != canonical_digest(body):
        raise ValueError('invalid map interface manifest')
    if actual != data.get('files'):
        changed = sorted(k for k in actual.keys() | data.get('files', {}).keys()
                         if actual.get(k) != data.get('files', {}).get(k))
        raise ValueError('active source/config inventory mismatch: ' + repr(changed))
    if parent.get('sha256') != canonical_digest({k:v for k,v in parent.items() if k != 'sha256'}):
        raise ValueError('invalid immutable map parent')
    if (data.get('parent_manifest_sha256') != hashlib.sha256(parent_path.read_bytes()).hexdigest()
            or data.get('parent_review_sha256') != parent['sha256']):
        raise ValueError('map parent manifest changed')
    baseline = json.loads((root/BASELINE/'baseline.json').read_text(encoding='utf-8'))
    archive = root/BASELINE/'source_and_evidence.zip'
    if (baseline.get('source_manifest_sha256') != parent['sha256']
            or baseline.get('archive_sha256') != hashlib.sha256(archive.read_bytes()).hexdigest()):
        raise ValueError('map baseline archive changed')
    evolution = json.loads((root/EVOLUTION).read_text(encoding='utf-8'))
    if evolution.get('parent_review_sha256') != parent['sha256'] or evolution.get('algorithm_changed') is not False:
        raise ValueError('invalid observation-only evolution identity')
    rows = evolution.get('changes', []); indexed = {row['path']: row for row in rows}
    required = {p for p in parent['files'].keys() | actual.keys()
                if parent['files'].get(p) != actual.get(p)} - {EVOLUTION}
    if len(rows) != len(indexed) or set(indexed) != required:
        raise ValueError('map evolution coverage mismatch')
    for name, row in indexed.items():
        if (row.get('before_sha256') != parent['files'].get(name)
                or row.get('after_sha256') != actual.get(name) or not row.get('reason')):
            raise ValueError('map evolution hash/reason mismatch: ' + name)
    return dict(review_sha256=data['sha256'], profile='active_map_interface_v1',
                inventory_sha256=canonical_digest(actual), parent_commit=parent['parent_head'], distribution=SCHEMA)
