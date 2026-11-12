"""Standalone additive admission, leaving existing provenance dispatch untouched."""
import hashlib
import ast
import json
import platform
from pathlib import Path
from project_paths import ROOT

MANIFEST = ROOT/'provenance/path_option_manifest_v4.json'


def legacy_digest(path):
    """Normalize ONLY the two exact pre-existing pure-MARL dispatch insertions."""
    path = Path(path)
    blocks = {
        'provenance.py': "if (root/'provenance/pure_marl_manifest.json').is_file():\n from chapter3_bser.baselines.source import verify\n return verify(root,source_files(root),canonical_digest)",
        'verify_source.py': "if (root/'provenance/pure_marl_manifest.json').is_file():\n from tools.verify_pure_marl import verify as verify_pure_marl\n return verify_pure_marl(root)"}
    if path.name not in blocks:
        return sha(path)
    expected = ast.dump(ast.parse(blocks[path.name]).body[0], include_attributes=False)
    class PreserveUserDispatch(ast.NodeTransformer):
        def visit_If(self, node):
            if ast.dump(node, include_attributes=False) == expected:
                return None
            return self.generic_visit(node)
    tree = PreserveUserDispatch().visit(ast.parse(path.read_text(encoding='utf-8')))
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def identity():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if digest({k: v for k, v in manifest.items() if k != 'sha256'}) != manifest['sha256']:
        raise ValueError('invalid independent manifest')
    for name, expected in manifest['files'].items():
        actual = legacy_digest(ROOT/name) if name in manifest['normalized_existing_dispatch'] else sha(ROOT/name)
        if actual != expected:
            raise ValueError('source differs from reviewed path-option manifest: '+name)
    return dict(schema='auv.path_option.source.v1', sha256=manifest['sha256'], parent_commit=manifest['parent_commit'])


def admission(validation):
    current = identity()
    r = json.loads(Path(validation).read_text(encoding='utf-8'))
    if platform.system() != 'Linux' or r.get('platform') != 'Linux' or r.get('passed') is not True:
        raise ValueError('successful actual Linux validation required')
    if r.get('source') != current or not r.get('native_smoke_passed'):
        raise ValueError('current source and explicit bounded native evidence required')
    if r.get('environment_initializations') != 2 or not 0 < r.get('physical_steps', 0) <= 64:
        raise ValueError('incomplete bounded native ledger')
    if not all(f'episode_{i:02d}.json' in r['artifacts'] for i in (1, 2)):
        raise ValueError('native episode evidence missing')
    for name, expected in r['artifacts'].items():
        if sha(Path(validation).parent/name) != expected:
            raise ValueError('validation evidence changed')
    return current


def new_output(path):
    result = Path(path).resolve()
    if not result.is_relative_to(ROOT/'work/collision_terminal'):
        raise ValueError('fresh output under work/collision_terminal required')
    if result.exists():
        raise FileExistsError(result)
    return result


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
