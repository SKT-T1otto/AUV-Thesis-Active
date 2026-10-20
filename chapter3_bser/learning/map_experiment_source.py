"""Verify the native-map training successor without rewriting map-core evidence."""
import hashlib
import json
from pathlib import Path

MANIFEST='provenance/map_experiment_manifest.json'
EVOLUTION='provenance/map_experiment_evolution.json'
BASELINE='provenance/baselines/map_core_20261020_v1'
SCHEMA='auv.active_source.map_experiment.v1'


def verify(root,actual,canonical_digest):
    root=Path(root);read=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
    manifest=read(MANIFEST);parent=read('provenance/map_core_manifest.json');baseline=read(BASELINE+'/baseline.json')
    if manifest.get('schema')!=SCHEMA or manifest.get('sha256')!=canonical_digest({k:v for k,v in manifest.items() if k!='sha256'}):raise ValueError('invalid map experiment manifest')
    if actual!=manifest['files']:raise ValueError('active source/config inventory mismatch')
    if (manifest.get('parent_review_sha256')!=parent['sha256']
            or manifest.get('parent_manifest_sha256')!=hashlib.sha256((root/'provenance/map_core_manifest.json').read_bytes()).hexdigest()
            or parent['sha256']!=canonical_digest({k:v for k,v in parent.items() if k!='sha256'})
            or baseline['source_identity']['review_sha256']!=parent['sha256']
            or baseline['archive_sha256']!=hashlib.sha256((root/BASELINE/'source_and_evidence.zip').read_bytes()).hexdigest()):raise ValueError('immutable map core parent changed')
    evo=read(EVOLUTION);rows=evo['changes'];indexed={r['path']:r for r in rows}
    required={p for p in parent['files'].keys()|actual.keys() if parent['files'].get(p)!=actual.get(p)}-{EVOLUTION}
    if len(indexed)!=len(rows) or set(indexed)!=required or evo.get('parent_review_sha256')!=parent['sha256']:raise ValueError('map experiment evolution coverage mismatch')
    for name,row in indexed.items():
        if row.get('before_sha256')!=parent['files'].get(name) or row.get('after_sha256')!=actual.get(name) or not row.get('reason'):raise ValueError('map experiment evolution hash mismatch')
    return dict(review_sha256=manifest['sha256'],profile='active_map_experiment_v1',inventory_sha256=canonical_digest(actual),
                parent_commit=baseline['source_identity']['parent_commit'],distribution=SCHEMA)
