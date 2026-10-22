"""Independent cache-only source/evolution and immutable-parent verification."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile
from tools.map_interface_source import digest, legacy_ast

MANIFEST = 'provenance/map_cache_manifest.json'
EVOLUTION = 'provenance/map_cache_evolution.json'
BASELINE = 'provenance/baselines/map_persistence_20261022_v1'
PARENT = 'provenance/map_persistence_manifest.json'


def inventory(root):
    root = Path(root)
    actual = {}
    for folder in ('core','chapter3_bser','configs','checks','tools','provenance/parent','provenance/history','provenance/baselines'):
        for path in (root/folder).rglob('*'):
            if path.is_file() and path.suffix in ('.py','.json','.zip'):
                actual[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (root/'scripts').rglob('*.sh'):
        actual[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('cli.py','project_paths.py','requirements.txt','AGENTS.md','.gitattributes','.gitignore',
                 'provenance/rebuild_evolution.json','provenance/map_stage1_evolution.json','provenance/map_stage1_manifest.json',
                 'provenance/map_core_evolution.json','provenance/map_core_manifest.json','provenance/map_experiment_evolution.json',
                 'provenance/map_experiment_manifest.json','provenance/map_convergence_evolution.json',
                 'provenance/map_convergence_manifest.json','provenance/map_persistence_evolution.json',PARENT,EVOLUTION):
        actual[name] = hashlib.sha256((root/name).read_bytes()).hexdigest()
    return actual


def method(source, class_name, name):
    tree = ast.parse(source)
    owner = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name == class_name)
    return ast.dump(next(node for node in owner.body if isinstance(node,ast.FunctionDef) and node.name == name), include_attributes=False)


def compatibility_kernel(source, *, prepared):
    function = next(node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name == 'compatible')
    start = next(i for i,node in enumerate(function.body) if isinstance(node,ast.Assign)
                 and any(isinstance(target,ast.Name) and target.id == 'best' for target in node.targets))
    tree = ast.Module(body=function.body[start:],type_ignores=[])
    if prepared:
        replacements = {
            'parallel': 'parallel=abs(direction)<=1e-12',
            'observed_hit': 'observed_hit=np.asarray(scan.hit_mask).reshape(-1)',
            'observed': 'observed=np.asarray(scan.distances).reshape(-1)',
        }
        class RestorePreparationReads(ast.NodeTransformer):
            def visit_Assign(self,node):
                if len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in replacements:
                    return ast.parse(replacements[node.targets[0].id]).body[0]
                return self.generic_visit(node)
        tree = RestorePreparationReads().visit(tree)
    return ast.dump(tree,include_attributes=False)


def verify(root):
    root = Path(root)
    read = lambda name: json.loads((root/name).read_text(encoding='utf-8'))
    sha = lambda name: hashlib.sha256((root/name).read_bytes()).hexdigest()
    data, parent, base = read(MANIFEST), read(PARENT), read(BASELINE+'/baseline.json')
    actual = inventory(root)
    if data.get('schema') != 'auv.active_source.map_cache.v1' or data.get('sha256') != digest({k:v for k,v in data.items() if k != 'sha256'}):
        raise ValueError('invalid independent cache manifest')
    if actual != data['files']:
        raise ValueError('independent source inventory mismatch')
    if (data['parent_review_sha256'] != parent['sha256'] or data['parent_manifest_sha256'] != sha(PARENT)
            or parent['sha256'] != digest({k:v for k,v in parent.items() if k != 'sha256'})
            or base['source_identity']['review_sha256'] != parent['sha256']
            or base['archive_sha256'] != sha(BASELINE+'/source_and_evidence.zip')):
        raise ValueError('immutable persistence parent changed')
    with zipfile.ZipFile(root/BASELINE/'source_and_evidence.zip') as archive:
        if set(archive.namelist()) != set(base['files']):
            raise ValueError('parent archive inventory changed')
        for name, expected in base['files'].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('frozen parent member changed')
            if name.startswith(('provenance/','verification/')) and sha(name) != expected:
                raise ValueError('parent evidence changed: '+name)
        for name, expected in parent['files'].items():
            if base['files'].get(name) != expected:
                raise ValueError('frozen parent source mismatch')
        for name in ('core/env/uav_env.py','core/env/mission_env.py','chapter3_bser/learning/base_runtime.py'):
            if legacy_ast(archive.read(name).decode()) != legacy_ast((root/name).read_text(encoding='utf-8')):
                raise ValueError('legacy AST changed: '+name)
        geometry = 'chapter3_bser/experiments/coupled_search_v1/geometry_hypotheses.py'
        belief = 'chapter3_bser/experiments/coupled_search_v1/coupled_belief.py'
        if compatibility_kernel(archive.read(geometry).decode(),prepared=False) != compatibility_kernel((root/geometry).read_text(encoding='utf-8'),prepared=True):
            raise ValueError('protected compatibility kernel changed')
        for class_name, name in (('RayConstraints','entry'),('RayConstraints','allowed'),
                                 ('GeometryHypothesis','free'),('GeometryHypothesis','identity'),('GeometryHypothesis','obstacles')):
            if method(archive.read(geometry).decode(),class_name,name) != method((root/geometry).read_text(encoding='utf-8'),class_name,name):
                raise ValueError('protected geometry kernel changed: '+name)
        for name in ('_prior','_propagate','_rebuild','consume','forecast','identity','initialize_hypotheses'):
            if method(archive.read(belief).decode(),'CoupledBelief',name) != method((root/belief).read_text(encoding='utf-8'),'CoupledBelief',name):
                raise ValueError('protected inference operation changed: '+name)
    permitted = {geometry, belief, 'chapter3_bser/learning/provenance.py',
        'tools/verify_source.py','tools/validate_checkout.py','checks/test_map_source.py','configs/rebuild_check_contract.json'}
    changed = {name for name, expected in parent['files'].items() if actual.get(name) != expected}
    if changed - permitted:
        raise ValueError('protected algorithm source changed: '+repr(sorted(changed-permitted)))
    new_runtime = {name for name in actual.keys()-parent['files'].keys() if name.startswith(('core/','chapter3_bser/'))}
    if new_runtime != {'chapter3_bser/experiments/coupled_search_v1/geometry_cache.py','chapter3_bser/learning/map_cache_source.py'}:
        raise ValueError('unexpected runtime extension')
    evolution = read(EVOLUTION); rows = evolution['changes']; indexed = {row['path']:row for row in rows}
    required = {name for name in parent['files'].keys() | actual.keys() if parent['files'].get(name) != actual.get(name)} - {EVOLUTION}
    if len(rows) != len(indexed) or set(indexed) != required or evolution['parent_review_sha256'] != parent['sha256']:
        raise ValueError('independent cache evolution coverage mismatch')
    for name, row in indexed.items():
        if row.get('before_sha256') != parent['files'].get(name) or row.get('after_sha256') != actual.get(name) or not row.get('reason'):
            raise ValueError('independent cache evolution hash mismatch: '+name)
    return dict(passed=True, files=len(actual), manifest_sha256=data['sha256'],
                baseline_archive_verified=True, learner_and_physics_byte_identical=True,
                all_parent_evidence_immutable=True, original_geometry_kernel_ast_identical=True,
                inference_replay_and_rng_operations_ast_identical=True, map_algorithm_revision='coupled_search_v2',
                compute_revision='exact_geometry_cache_v1')
