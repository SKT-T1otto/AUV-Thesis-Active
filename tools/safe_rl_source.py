"""Independent additive-source verifier. Archived parent source is DATA only."""
import ast,hashlib,json,zipfile
from pathlib import Path
from tools.executor_diagnostics_source import inventory as parent_inventory
from tools.map_interface_source import digest
MANIFEST='provenance/safe_rl_manifest.json'
EVOLUTION='provenance/safe_rl_evolution.json'
PARENT='provenance/executor_diagnostics_manifest.json'
BASELINE='provenance/baselines/executor_diagnostics_20261027_v1'

def inventory(root,*,allow_missing_evolution=False):
    root=Path(root);actual=parent_inventory(root);actual[PARENT]=hashlib.sha256((root/PARENT).read_bytes()).hexdigest()
    if not allow_missing_evolution or (root/EVOLUTION).exists():actual[EVOLUTION]=hashlib.sha256((root/EVOLUTION).read_bytes()).hexdigest()
    return actual

class RemoveSafetyMetadata(ast.NodeTransformer):
    def visit_If(self,node):
        if any(isinstance(n,ast.Constant) and n.value==MANIFEST for n in ast.walk(node.test)):return None
        return self.generic_visit(node)
    def visit_Dict(self,node):
        pairs=[(k,v) for k,v in zip(node.keys,node.values) if not isinstance(k,ast.Constant) or k.value!='active_safe_rl_v1']
        node.keys=[k for k,v in pairs];node.values=[v for k,v in pairs];return self.generic_visit(node)
    def visit_Tuple(self,node):
        node.elts=[n for n in node.elts if not isinstance(n,ast.Constant) or n.value!='active_safe_rl_v1'];return self.generic_visit(node)

def normalized_baseline(source):
    tree=ast.parse(source);baseline=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BaselineMissionRuntime')
    method=next((n for n in baseline.body if isinstance(n,ast.FunctionDef) and n.name=='_make_environment'),None)
    if method is not None:
        expected=ast.parse('class X:\n @staticmethod\n def _make_environment(config):\n  return _make_base_env(config)').body[0].body[0]
        if ast.dump(method,include_attributes=False)!=ast.dump(expected,include_attributes=False):raise ValueError('baseline factory default changed')
        baseline.body.remove(method);init=next(n for n in baseline.body if isinstance(n,ast.FunctionDef) and n.name=='__init__');replacements=0
        for node in init.body:
            if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='_make_environment':
                if ast.dump(node.value,include_attributes=False)!=ast.dump(ast.parse('self._make_environment(config)',mode='eval').body,include_attributes=False):raise ValueError('invalid factory dispatch')
                node.value=ast.parse('_make_base_env(config)',mode='eval').body;replacements+=1
        if replacements!=1:raise ValueError('missing baseline factory dispatch')
    return ast.dump(tree,include_attributes=False)

def verify(root):
    root=Path(root);read=lambda n:json.loads((root/n).read_text(encoding='utf-8'));sha=lambda n:hashlib.sha256((root/n).read_bytes()).hexdigest()
    actual=inventory(root);data=read(MANIFEST);parent=read(PARENT);base=read(BASELINE+'/baseline.json')
    if data.get('schema')!='auv.active_source.safe_rl.v1' or data.get('sha256')!=digest({k:v for k,v in data.items() if k!='sha256'}):raise ValueError('invalid safety-RL manifest')
    if actual!=data['files']:raise ValueError('independent source inventory mismatch')
    if data.get('algorithm_changed') is not True or data['parent_review_sha256']!=parent['sha256'] or data['parent_manifest_sha256']!=sha(PARENT):raise ValueError('immutable safety parent changed')
    if base['source_identity']['review_sha256']!=parent['sha256'] or sha(BASELINE+'/source_and_docs.zip')!=base['archive_sha256']:raise ValueError('immutable parent archive changed')
    with zipfile.ZipFile(root/BASELINE/'source_and_docs.zip') as archive:
        external=base['external_immutable_files']
        if set(archive.namelist())!=set(base['files'])-set(external):raise ValueError('parent archive inventory changed')
        for name,h in base['files'].items():
            if name in external and sha(name)!=h:raise ValueError('parent evidence changed: '+name)
            if name not in external and hashlib.sha256(archive.read(name)).hexdigest()!=h:raise ValueError('parent archive member changed: '+name)
        for name,h in parent['files'].items():
            if not name.startswith(('core/','chapter3_bser/')) or actual.get(name)==h:continue
            if name=='chapter3_bser/learning/base_runtime.py':
                if normalized_baseline((root/name).read_text(encoding='utf-8'))!=normalized_baseline(archive.read(name).decode('utf-8')):raise ValueError('baseline runtime changed beyond factory hook')
                continue
            if name=='chapter3_bser/learning/provenance.py':
                new=RemoveSafetyMetadata().visit(ast.parse((root/name).read_text(encoding='utf-8')))
                if ast.dump(new,include_attributes=False)!=ast.dump(ast.parse(archive.read(name).decode('utf-8')),include_attributes=False):raise ValueError('source admission changed beyond safety metadata')
                continue
            if name.startswith('core/env/'):raise ValueError('legacy AST changed: '+name)
            if name.endswith('geometry_hypotheses.py'):raise ValueError('protected geometry kernel changed: '+name)
            if name.endswith('bounded_belief.py'):raise ValueError('observer changed algorithm AST: '+name)
            raise ValueError('protected baseline runtime changed: '+name)
    additions={n for n in actual.keys()-parent['files'].keys() if n.startswith(('core/','chapter3_bser/'))}
    if any(not (n.startswith('chapter3_bser/safe_rl/') or n=='chapter3_bser/learning/safe_rl_source.py') for n in additions):raise ValueError('unexpected runtime extension')
    forbidden={'ground_truth_obstacles','obstacles','_raycast_ground_truth','_nearest_obstacle_distance','_flow_at','_flow_phase_x','_flow_phase_y','_obstacle_lower','_obstacle_upper'}
    for name in actual:
        if not name.startswith(('core/','chapter3_bser/')) or not name.endswith('.py'):continue
        tree=ast.parse((root/name).read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            imports=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
            if any(n=='tools' or n.startswith('tools.') for n in imports):raise ValueError('tool imported by runtime: '+name)
            if name.startswith('chapter3_bser/safe_rl/') and isinstance(node,ast.Attribute) and node.attr in forbidden:raise ValueError('hidden world queried by safety: '+name)
    evo=read(EVOLUTION);indexed={r['path']:r for r in evo['changes']};required={n for n in parent['files'].keys()|actual.keys() if parent['files'].get(n)!=actual.get(n)}-{EVOLUTION}
    if len(indexed)!=len(evo['changes']) or set(indexed)!=required or evo['parent_review_sha256']!=parent['sha256']:raise ValueError('safety evolution coverage mismatch')
    for name,row in indexed.items():
        if row.get('before_sha256')!=parent['files'].get(name) or row.get('after_sha256')!=actual.get(name) or not row.get('reason'):raise ValueError('safety evolution hash mismatch: '+name)
    return dict(passed=True,files=len(actual),manifest_sha256=data['sha256'],algorithm_changed=True,original_core_bytes_unchanged=True,original_d2_b2_b3_bytes_unchanged=True,baseline_runtime_changed_only_by_default_preserving_factory=True,all_parent_evidence_immutable=True,safety_hidden_world_attribute_audit_passed=True,scope='Separate OLD all-agent safety learning; no performance acceptance')
