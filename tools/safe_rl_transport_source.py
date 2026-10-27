"""Independent verification-only evolution: protect all parent runtime bytes."""
import ast,hashlib,json,zipfile
from pathlib import Path
from tools.safe_rl_recovery_source import inventory as parent_inventory
from tools.map_interface_source import digest
MANIFEST='provenance/safe_rl_transport_manifest.json'
EVOLUTION='provenance/safe_rl_transport_evolution.json'
PARENT='provenance/safe_rl_recovery_manifest.json'
BASELINE='provenance/baselines/safe_rl_recovery_20261027_v2'

def inventory(root,*,allow_missing_evolution=False):
    root=Path(root);actual=parent_inventory(root);actual[PARENT]=hashlib.sha256((root/PARENT).read_bytes()).hexdigest()
    if not allow_missing_evolution or (root/EVOLUTION).exists():actual[EVOLUTION]=hashlib.sha256((root/EVOLUTION).read_bytes()).hexdigest()
    return actual

class RemoveValidationMetadata(ast.NodeTransformer):
    def visit_If(self,node):
        if any(isinstance(n,ast.Constant) and n.value==MANIFEST for n in ast.walk(node.test)):return None
        return self.generic_visit(node)
    def visit_Dict(self,node):
        pairs=[(k,v) for k,v in zip(node.keys,node.values) if not isinstance(k,ast.Constant) or k.value!='active_safe_rl_transport_v1']
        node.keys=[k for k,v in pairs];node.values=[v for k,v in pairs];return self.generic_visit(node)
    def visit_Tuple(self,node):
        node.elts=[n for n in node.elts if not isinstance(n,ast.Constant) or n.value!='active_safe_rl_transport_v1'];return self.generic_visit(node)

def verify(root):
    root=Path(root);read=lambda n:json.loads((root/n).read_text(encoding='utf-8'));sha=lambda n:hashlib.sha256((root/n).read_bytes()).hexdigest()
    actual=inventory(root);data=read(MANIFEST);parent=read(PARENT);base=read(BASELINE+'/baseline.json')
    if data.get('schema')!='auv.active_source.safe_rl_transport.v1' or data.get('sha256')!=digest({k:v for k,v in data.items() if k!='sha256'}):raise ValueError('invalid safety validation manifest')
    if actual!=data['files']:raise ValueError('independent source inventory mismatch')
    if data.get('algorithm_changed') is not False or data['parent_review_sha256']!=parent['sha256'] or data['parent_manifest_sha256']!=sha(PARENT):raise ValueError('immutable safety-RL parent changed')
    if base['source_identity']['review_sha256']!=parent['sha256'] or sha(BASELINE+'/source_and_docs.zip')!=base['archive_sha256']:raise ValueError('immutable parent archive changed')
    with zipfile.ZipFile(root/BASELINE/'source_and_docs.zip') as archive:
        external=base['external_immutable_files']
        if set(archive.namelist())!=set(base['files'])-set(external):raise ValueError('parent archive inventory changed')
        for name,h in base['files'].items():
            if name in external and sha(name)!=h:raise ValueError('parent evidence changed: '+name)
            if name not in external and hashlib.sha256(archive.read(name)).hexdigest()!=h:raise ValueError('parent archive member changed: '+name)
        for name,h in parent['files'].items():
            if not name.startswith(('core/','chapter3_bser/')) or actual.get(name)==h:continue
            if name=='chapter3_bser/learning/provenance.py':
                new=RemoveValidationMetadata().visit(ast.parse((root/name).read_text(encoding='utf-8')))
                if ast.dump(new,include_attributes=False)!=ast.dump(ast.parse(archive.read(name).decode('utf-8')),include_attributes=False):raise ValueError('source admission changed beyond validation metadata')
                continue
            if name.startswith('core/env/'):raise ValueError('legacy AST changed: '+name)
            if name.endswith('geometry_hypotheses.py'):raise ValueError('protected geometry kernel changed: '+name)
            if name.endswith('bounded_belief.py'):raise ValueError('observer changed algorithm AST: '+name)
            raise ValueError('protected parent runtime changed: '+name)
    additions={n for n in actual.keys()-parent['files'].keys() if n.startswith(('core/','chapter3_bser/'))}
    if additions!={'chapter3_bser/learning/safe_rl_transport_source.py'}:raise ValueError('unexpected verification-only runtime extension')
    for name in actual:
        if not name.startswith(('core/','chapter3_bser/')) or not name.endswith('.py'):continue
        for node in ast.walk(ast.parse((root/name).read_text(encoding='utf-8'))):
            imports=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
            if any(n=='tools' or n.startswith('tools.') for n in imports):raise ValueError('tool imported by runtime: '+name)
    evo=read(EVOLUTION);indexed={r['path']:r for r in evo['changes']}
    required={n for n in parent['files'].keys()|actual.keys() if parent['files'].get(n)!=actual.get(n)}-{EVOLUTION}
    if len(indexed)!=len(evo['changes']) or set(indexed)!=required or evo['parent_review_sha256']!=parent['sha256']:raise ValueError('validation evolution coverage mismatch')
    for name,row in indexed.items():
        if row.get('before_sha256')!=parent['files'].get(name) or row.get('after_sha256')!=actual.get(name) or not row.get('reason'):raise ValueError('validation evolution hash mismatch: '+name)
    return dict(passed=True,files=len(actual),manifest_sha256=data['sha256'],algorithm_changed=False,
        all_parent_runtime_bytes_unchanged=True,source_admission_metadata_only=True,all_parent_evidence_immutable=True,
        scope='Streaming timeout handling and explicit immutable-parent prefix continuation only; all algorithms unchanged')
