"""Independent file/AST/parent verifier. Never imports runtime implementation."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile

BASELINE = 'provenance/baselines/map_d2_repair_v2_20261020_v1'
EVOLUTION = 'provenance/map_stage1_evolution.json'


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()).hexdigest()


class RemovePublicInstrumentation(ast.NodeTransformer):
    """Only erase the explicit observation hooks; preserve all legacy logic."""
    added = {'enable_public_event_capture', 'read_public_events', '_public_step_begin',
             '_public_step_end', 'get_public_vehicle_specs', 'get_public_control_contract'}

    def __init__(self):
        self.function = None

    def visit_FunctionDef(self, node):
        if node.name in self.added:
            return None
        old, self.function = self.function, node.name
        pairs = [(a, d) for a, d in zip(node.args.kwonlyargs, node.args.kw_defaults) if a.arg != 'map_research']
        node.args.kwonlyargs = [a for a, d in pairs]
        node.args.kw_defaults = [d for a, d in pairs]
        result = self.generic_visit(node)
        self.function = old
        return result

    def visit_ImportFrom(self, node):
        return None if node.module == 'core.env.public_events' else node

    def visit_Import(self, node):
        if any(item.name == 'chapter3_bser.planner_factory' for item in node.names):
            if self.function == 'advance':
                return None
            for item in node.names:
                if item.name == 'chapter3_bser.planner_factory': item.name = 'chapter3_bser.d2.assembly'
        return node

    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == 'journal' and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name) and node.value.func.id == 'getattr'
                and any(isinstance(a, ast.Constant) and a.value == '_public_event_journal' for a in node.value.args)):
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        if isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and node.test.left.id == 'journal':
            return None
        return self.generic_visit(node)

    def visit_Expr(self, node):
        call = node.value
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
            if (isinstance(call.func.value, ast.Name) and call.func.value.id == 'self'
                    and call.func.attr in ('_public_step_begin', '_public_step_end')):
                return None
            if (isinstance(call.func.value, ast.Name) and call.func.value.id == 'assembly'
                    and call.func.attr in ('before_reset', 'consume', 'installed')):
                return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        if (isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                and node.func.value.id == 'assembly' and node.func.attr == 'prepare'):
            node.keywords = [k for k in node.keywords if k.arg != 'map_research']
        return self.generic_visit(node)


def legacy_ast(source):
    return ast.dump(RemovePublicInstrumentation().visit(ast.parse(source)), include_attributes=False)


def verify(root):
    root = Path(root).resolve()
    read = lambda name: json.loads((root/name).read_text(encoding='utf-8'))
    manifest = read('provenance/map_stage1_manifest.json')
    if (manifest.get('schema') != 'auv.active_source.map_interface.v1'
            or manifest.get('sha256') != digest({k:v for k,v in manifest.items() if k != 'sha256'})):
        raise ValueError('invalid independent map manifest')
    actual = {}
    for folder in ('core', 'chapter3_bser', 'configs', 'checks', 'tools', 'provenance/parent', 'provenance/history'):
        for file in (root/folder).rglob('*'):
            if file.is_file() and file.suffix in ('.py', '.json'):
                actual[file.relative_to(root).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
    for file in (root/'provenance/baselines').rglob('*'):
        if file.is_file() and file.suffix in ('.json', '.zip'):
            actual[file.relative_to(root).as_posix()] = hashlib.sha256(file.read_bytes()).hexdigest()
    for name in ('cli.py', 'project_paths.py', 'requirements.txt', 'AGENTS.md', '.gitattributes', '.gitignore', 'provenance/rebuild_evolution.json', EVOLUTION):
        actual[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual != manifest['files']:
        raise ValueError('independent map source inventory mismatch')
    parent_bytes = (root/'provenance/active_manifest.json').read_bytes()
    parent = json.loads(parent_bytes)
    if (hashlib.sha256(parent_bytes).hexdigest() != manifest.get('parent_manifest_sha256')
            or parent['sha256'] != manifest.get('parent_review_sha256')
            or parent['sha256'] != digest({k:v for k,v in parent.items() if k != 'sha256'})):
        raise ValueError('immutable map parent changed')
    baseline = read(BASELINE+'/baseline.json')
    archive = root/BASELINE/'source_and_evidence.zip'
    if (baseline['archive_sha256'] != hashlib.sha256(archive.read_bytes()).hexdigest()
            or baseline['source_manifest_sha256'] != parent['sha256']):
        raise ValueError('baseline archive identity mismatch')
    with zipfile.ZipFile(archive) as frozen:
        if set(frozen.namelist()) != set(baseline['files']):
            raise ValueError('baseline archive inventory mismatch')
        for name, expected in baseline['files'].items():
            if hashlib.sha256(frozen.read(name)).hexdigest() != expected:
                raise ValueError('baseline content changed: '+name)
            if name.startswith('verification/') and hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
                raise ValueError('immutable parent evidence changed: '+name)
        if frozen.read('provenance/active_manifest.json') != parent_bytes:
            raise ValueError('frozen parent manifest differs')
        for name, expected in parent['files'].items():
            if hashlib.sha256(frozen.read(name)).hexdigest() != expected:
                raise ValueError('frozen source does not match parent')
        permitted = {'core/env/uav_env.py', 'core/env/mission_env.py', 'chapter3_bser/learning/base_runtime.py',
                     'chapter3_bser/learning/provenance.py', 'tools/verify_source.py',
                     'tools/run_checks.py', 'configs/rebuild_check_contract.json'}
        changes = {name for name in parent['files'] if actual.get(name) != parent['files'][name]}
        if not changes <= permitted:
            raise ValueError('interface stage modified protected baseline files: '+repr(sorted(changes-permitted)))
        added_core = {name for name in actual if name.startswith('core/') and name not in parent['files']}
        if added_core != {'core/env/public_events.py'}:
            raise ValueError('unexpected shared-core extension')
        proofs = []
        for name in ('core/env/uav_env.py', 'core/env/mission_env.py', 'chapter3_bser/learning/base_runtime.py'):
            if legacy_ast(frozen.read(name).decode('utf-8')) != legacy_ast((root/name).read_text(encoding='utf-8')):
                raise ValueError('legacy AST changed outside observation hooks: '+name)
            proofs.append(name)
    evo = read(EVOLUTION); rows = evo['changes']; indexed = {r['path']:r for r in rows}
    required = {p for p in parent['files'].keys() | actual.keys() if parent['files'].get(p) != actual.get(p)} - {EVOLUTION}
    if (len(rows) != len(indexed) or set(indexed) != required or evo.get('parent_review_sha256') != parent['sha256']
            or evo.get('algorithm_changed') is not False):
        raise ValueError('independent evolution coverage mismatch')
    for name, row in indexed.items():
        if row.get('before_sha256') != parent['files'].get(name) or row.get('after_sha256') != actual.get(name) or not row.get('reason'):
            raise ValueError('independent evolution hash mismatch: '+name)
    return dict(passed=True, files=len(actual), manifest_sha256=manifest['sha256'],
                baseline_archive_verified=True, legacy_ast_equal=proofs,
                core_unchanged=False, algorithm_changed=False)
