import json
import platform
from pathlib import Path
from project_paths import ROOT
from chapter3_bser.path_option.source import sha,digest,legacy_digest,write,new_output

MANIFEST=ROOT/'provenance/r123_governed_manifest_v2.json'


def identity():
    m=json.loads(MANIFEST.read_text(encoding='utf-8'))
    if digest({k:v for k,v in m.items() if k!='sha256'})!=m['sha256']:raise ValueError('invalid repair manifest')
    for name,expected in m['files'].items():
        actual=legacy_digest(ROOT/name) if name in m['normalized_existing_dispatch'] else sha(ROOT/name)
        if actual!=expected:raise ValueError('repair source changed: '+name)
    return dict(schema='auv.r123.governed.source.v1',sha256=m['sha256'],parent_commit=m['parent_commit'])


def admission(path):
    p=Path(path);r=json.loads(p.read_text(encoding='utf-8'))
    if platform.system()!='Linux' or r.get('platform')!='Linux' or not r.get('passed'):
        raise ValueError('successful actual Linux repair validation required')
    if r.get('source')!=identity() or not r.get('native_smoke_passed'):
        raise ValueError('current repair source native smoke required; old R123 receipts cannot be reused')
    if r.get('environment_initializations')!=2 or not 0<r.get('physical_steps',0)<=64 or r.get('native_training_updates')!=0:
        raise ValueError('two-case bounded native ledger required')
    if not all(f'episode_{i:02d}.json' in r['artifacts'] for i in range(2)):raise ValueError('native episodes missing')
    for name,expected in r['artifacts'].items():
        if sha(p.parent/name)!=expected:raise ValueError('validation artifact changed: '+name)
    return r['source']
