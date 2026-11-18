"""Independent evolution identity; immutable tracking parent remains valid."""
import json
from project_paths import ROOT
from chapter3_bser.path_option.source import sha,digest
from chapter3_bser.r123_tracking.source import identity as parent_identity

MANIFEST=ROOT/'provenance/r123_task_sync_manifest_v1.json'


def identity():
    m=json.loads(MANIFEST.read_text(encoding='utf-8'))
    if digest({k:v for k,v in m.items() if k!='sha256'})!=m['sha256']:raise ValueError('invalid sync manifest')
    if m['parent_source']!=parent_identity():raise ValueError('tracking parent changed')
    for n,h in m['files'].items():
        if sha(ROOT/n)!=h:raise ValueError('sync source changed: '+n)
    return dict(schema='auv.r123.task_sync.source.v1',sha256=m['sha256'],parent_source=m['parent_source'])
