"""Independent observer-only AST parity and immutable parent verification."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile
from tools.map_route_health_source import inventory as parent_inventory
from tools.map_interface_source import digest,legacy_ast

MANIFEST='provenance/map_diagnostics_manifest.json'
EVOLUTION='provenance/map_diagnostics_evolution.json'
PARENT='provenance/map_route_health_manifest.json'
BASELINE='provenance/baselines/map_route_health_20261025_v1'


def inventory(root):
    root=Path(root);actual=parent_inventory(root)
    for name in (PARENT,EVOLUTION):actual[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    return actual


class RemoveObservers(ast.NodeTransformer):
    def observed(self,node):
        return any(isinstance(n,ast.Attribute) and n.attr in ('inference_diagnostics','planning_diagnostics','activity') for n in ast.walk(node))

    def visit_ImportFrom(self,node):
        return None if node.module=='diagnostics' else node

    def visit_Expr(self,node):
        return None if self.observed(node) else self.generic_visit(node)

    def visit_Assign(self,node):
        return None if any(self.observed(t) for t in node.targets) else self.generic_visit(node)

    def visit_FunctionDef(self,node):
        node.decorator_list=[d for d in node.decorator_list if not (isinstance(d,ast.Name) and d.id in ('observe_inference','observe_allocation'))
                             and not (isinstance(d,ast.Call) and isinstance(d.func,ast.Name) and d.func.id=='observe_planning')]
        self.generic_visit(node)
        # The found-target branch stores its original return value only long
        # enough to observe it; normalize that adjacent assignment/return.
        if len(node.body)>=2 and isinstance(node.body[-2],ast.Assign) and isinstance(node.body[-1],ast.Return):
            assignment,ret=node.body[-2:]
            if len(assignment.targets)==1 and isinstance(assignment.targets[0],ast.Name) and isinstance(ret.value,ast.Name) and ret.value.id==assignment.targets[0].id:
                node.body[-2:]=[ast.Return(value=assignment.value)]
        return node


def algorithm_ast(source):
    return ast.dump(RemoveObservers().visit(ast.parse(source)),include_attributes=False)


def verify(root):
    root=Path(root);read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    sha=lambda n:hashlib.sha256((root/n).read_bytes()).hexdigest()
    actual=inventory(root);data=read(MANIFEST);parent=read(PARENT);base=read(BASELINE+'/baseline.json')
    if data.get('schema')!='auv.active_source.map_diagnostics.v1' or data.get('sha256')!=digest({k:v for k,v in data.items() if k!='sha256'}):raise ValueError('invalid map diagnostics manifest')
    if actual!=data['files']:raise ValueError('independent source inventory mismatch')
    if data['parent_review_sha256']!=parent['sha256'] or data['parent_manifest_sha256']!=sha(PARENT) or base['source_identity']['review_sha256']!=parent['sha256'] or sha(BASELINE+'/source_and_evidence.zip')!=base['archive_sha256']:
        raise ValueError('immutable route-health parent changed')
    instrumented={f'chapter3_bser/experiments/coupled_search_v1/{name}.py' for name in
                  ('coupled_belief','bounded_belief','allocator','session','controller','assembly')}
    allowed=instrumented|{'chapter3_bser/learning/map_contract.py','chapter3_bser/learning/map_report.py',
        'chapter3_bser/learning/bounded_experiment.py','chapter3_bser/learning/map_suite.py','chapter3_bser/learning/provenance.py',
        'checks/test_map_source.py','checks/test_route_health.py','configs/rebuild_check_contract.json',
        'tools/verify_source.py','tools/validate_checkout.py'}
    with zipfile.ZipFile(root/BASELINE/'source_and_evidence.zip') as archive:
        external=base['external_frozen_archives']
        if set(archive.namelist())!=set(base['files'])-set(external):raise ValueError('parent archive inventory changed')
        for name,h in base['files'].items():
            saved=sha(name) if name in external else hashlib.sha256(archive.read(name)).hexdigest()
            if saved!=h:raise ValueError('frozen parent member changed: '+name)
            if name.startswith(('provenance/','verification/')) and sha(name)!=h:raise ValueError('parent evidence changed: '+name)
        for name in ('core/env/uav_env.py','core/env/mission_env.py','chapter3_bser/learning/base_runtime.py'):
            if legacy_ast(archive.read(name).decode())!=legacy_ast((root/name).read_text(encoding='utf-8')):raise ValueError('legacy AST changed: '+name)
        for name in ('geometry_hypotheses','geometry_cache','target_particles','persistent_map','bounded_config','numerics'):
            path=f'chapter3_bser/experiments/coupled_search_v1/{name}.py'
            if (root/path).read_bytes()!=archive.read(path):raise ValueError('protected geometry kernel changed: '+name)
        for name in instrumented:
            if algorithm_ast(archive.read(name).decode())!=algorithm_ast((root/name).read_text(encoding='utf-8')):
                raise ValueError('observer changed algorithm AST: '+name)
        def health(source):
            tree=ast.parse(source);function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='metrics')
            return ast.dump(next(n for n in function.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and isinstance(n.test.left,ast.Subscript)),include_attributes=False)
        name='chapter3_bser/learning/map_contract.py'
        if health(archive.read(name).decode())!=health((root/name).read_text(encoding='utf-8')):raise ValueError('strict health predicates changed')
    changed={n for n,h in parent['files'].items() if actual.get(n)!=h}
    if changed-allowed:raise ValueError('protected algorithm source changed: '+repr(sorted(changed-allowed)))
    new_runtime={n for n in actual.keys()-parent['files'].keys() if n.startswith(('core/','chapter3_bser/'))}
    if new_runtime!={'chapter3_bser/experiments/coupled_search_v1/diagnostics.py','chapter3_bser/learning/map_diagnostics.py','chapter3_bser/learning/map_diagnostic_source.py'}:
        raise ValueError('unexpected runtime diagnostic extension')
    evo=read(EVOLUTION);indexed={r['path']:r for r in evo['changes']}
    required={n for n in parent['files'].keys()|actual.keys() if parent['files'].get(n)!=actual.get(n)}-{EVOLUTION}
    if len(indexed)!=len(evo['changes']) or set(indexed)!=required or evo['parent_review_sha256']!=parent['sha256']:raise ValueError('map diagnostics evolution coverage mismatch')
    for n,row in indexed.items():
        if row.get('before_sha256')!=parent['files'].get(n) or row.get('after_sha256')!=actual.get(n) or not row.get('reason'):raise ValueError('map diagnostics evolution hash mismatch: '+n)
    return dict(passed=True,files=len(actual),manifest_sha256=data['sha256'],baseline_archive_verified=True,
        all_parent_evidence_immutable=True,learner_and_physics_byte_identical=True,original_d2_byte_identical=True,
        algorithm_ast_parity=True,geometry_kernels_byte_identical=True,strict_health_predicates_preserved=True,
        map_algorithm_revision='coupled_search_bounded_v3',compute_revision='diagnostics_v1')
