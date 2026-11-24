"""Identify the reviewed successor without changing any parent source lock."""
from hashlib import sha256
import json
from project_paths import ROOT


LOCK = ROOT / 'provenance/launch_interventions_v1.json'


def checked_source():
    data = json.loads(LOCK.read_bytes())
    for name, expected in data['files'].items():
        if sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise ValueError('intervention source or input changed: ' + name)
    return sha256(LOCK.read_bytes()).hexdigest()


def reviewed_successor(name, old, *, normalized=False):
    data = json.loads(LOCK.read_bytes())
    field = 'parent_lf' if normalized else 'parent_raw'
    return data[field].get(name) == old and checked_source()
