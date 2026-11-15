"""Independent source admission: frozen legacy files, no old-weight relabeling."""
import json
import platform
from pathlib import Path
from project_paths import ROOT
from chapter3_bser.path_option.source import sha, digest, legacy_digest, write, new_output

MANIFEST=ROOT/'provenance/r123_manifest_v1.json'


def identity():
    m=json.loads(MANIFEST.read_text(encoding='utf-8'))
    if digest({k:v for k,v in m.items() if k!='sha256'})!=m['sha256']:
        raise ValueError('invalid R123 source manifest')
    for n,h in m['files'].items():
        actual=legacy_digest(ROOT/n) if n in m['normalized_existing_dispatch'] else sha(ROOT/n)
        if actual!=h:raise ValueError('R123 source changed: '+n)
    return dict(schema='auv.r123.source.v1',sha256=m['sha256'],parent_commit=m['parent_commit'])


def admission(path):
    r=json.loads(Path(path).read_text(encoding='utf-8'))
    if platform.system()!='Linux' or r.get('platform')!='Linux' or not r.get('passed'):
        raise ValueError('current actual Linux R123 validation required')
    if r['source']!=identity() or not r.get('native_smoke_passed'):
        raise ValueError('source-matched native R123 smoke required')
    if r['environment_initializations']!=4 or not 0<r['physical_steps']<=128:
        raise ValueError('complete bounded four-case ledger required')
    for name,h in r['artifacts'].items():
        if sha(Path(path).parent/name)!=h:raise ValueError('validation artifact changed')
    if not all(f'episode_{i:02d}.json' in r['artifacts'] for i in range(4)):
        raise ValueError('native episode evidence missing')
    return r['source']
