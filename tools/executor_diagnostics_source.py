"""Independent passive-only evolution verifier; archived source is never run."""
import ast
import hashlib
import json
from pathlib import Path
import zipfile
from tools.executor_recovery_source import inventory as parent_inventory
from tools.map_interface_source import digest

MANIFEST='provenance/executor_diagnostics_manifest.json'
EVOLUTION='provenance/executor_diagnostics_evolution.json'
PARENT='provenance/executor_recovery_manifest.json'
BASELINE='provenance/baselines/executor_recovery_20261026_v2'


def inventory(root, *, allow_missing_evolution=False):
    root=Path(root);actual=parent_inventory(root)
    actual[PARENT]=hashlib.sha256((root/PARENT).read_bytes()).hexdigest()
    if not allow_missing_evolution or (root/EVOLUTION).exists():
        actual[EVOLUTION]=hashlib.sha256((root/EVOLUTION).read_bytes()).hexdigest()
    return actual


class RemoveDiagnosticMetadata(ast.NodeTransformer):
    def visit_If(self,node):
        if any(isinstance(n,ast.Constant) and n.value=='provenance/executor_diagnostics_manifest.json' for n in ast.walk(node.test)):
            return None
        return self.generic_visit(node)

    def visit_Dict(self,node):
        pairs=[(k,v) for k,v in zip(node.keys,node.values) if not isinstance(k,ast.Constant) or k.value!='active_executor_diagnostics_v1']
        node.keys=[k for k,v in pairs];node.values=[v for k,v in pairs]
        return self.generic_visit(node)

    def visit_Tuple(self,node):
        node.elts=[n for n in node.elts if not isinstance(n,ast.Constant) or n.value!='active_executor_diagnostics_v1']
        return self.generic_visit(node)


def verify(root):
    root=Path(root);read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    sha=lambda n:hashlib.sha256((root/n).read_bytes()).hexdigest()
    actual,data,parent,base=inventory(root),read(MANIFEST),read(PARENT),read(BASELINE+'/baseline.json')
    if data.get('schema')!='auv.active_source.executor_diagnostics.v1' or data.get('sha256')!=digest({k:v for k,v in data.items() if k!='sha256'}):
        raise ValueError('invalid diagnostics manifest')
    if actual!=data['files']:
        raise ValueError('independent source inventory mismatch')
    if data.get('algorithm_changed') is not False or data['parent_review_sha256']!=parent['sha256'] or data['parent_manifest_sha256']!=sha(PARENT):
        raise ValueError('immutable diagnostics parent changed')
    if base['source_identity']['review_sha256']!=parent['sha256'] or sha(BASELINE+'/source_and_docs.zip')!=base['archive_sha256']:
        raise ValueError('immutable recovery archive changed')
    with zipfile.ZipFile(root/BASELINE/'source_and_docs.zip') as archive:
        external=base['external_immutable_files']
        if set(archive.namelist())!=set(base['files'])-set(external):
            raise ValueError('parent archive inventory changed')
        for name,h in base['files'].items():
            if name in external and sha(name)!=h:
                raise ValueError('parent evidence changed: '+name)
            if name not in external and hashlib.sha256(archive.read(name)).hexdigest()!=h:
                raise ValueError('frozen parent member changed: '+name)
        for name,h in parent['files'].items():
            if name.startswith(('core/','chapter3_bser/')) and name!='chapter3_bser/learning/provenance.py' and actual.get(name)!=h:
                if name.startswith('core/env/'):
                    raise ValueError('legacy AST changed: '+name)
                if name.endswith('geometry_hypotheses.py'):
                    raise ValueError('protected geometry kernel changed: '+name)
                if name.endswith('bounded_belief.py'):
                    raise ValueError('observer changed algorithm AST: '+name)
                raise ValueError('protected runtime source changed: '+name)
        old=ast.parse(archive.read('chapter3_bser/learning/provenance.py').decode('utf-8'))
        new=ast.parse((root/'chapter3_bser/learning/provenance.py').read_text(encoding='utf-8'))
        functions=lambda tree:{n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
        a,b=functions(old),functions(new)
        for name in ('canonical_digest','require_review','training_admission'):
            if a[name]!=b[name]:
                raise ValueError('protected source admission changed: '+name)
        if ast.dump(RemoveDiagnosticMetadata().visit(new),include_attributes=False)!=ast.dump(old,include_attributes=False):
            raise ValueError('source admission changed beyond diagnostic metadata')
    additions={n for n in actual.keys()-parent['files'].keys() if n.startswith(('core/','chapter3_bser/'))}
    if additions!={'chapter3_bser/learning/executor_diagnostics_source.py'}:
        raise ValueError('unexpected passive runtime extension')
    # Observers must never be imported by production behavior modules.
    for name in actual:
        if name.startswith(('core/','chapter3_bser/')) and name.endswith('.py'):
            tree=ast.parse((root/name).read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                imports=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
                if any(n=='tools' or n.startswith('tools.') for n in imports):
                    raise ValueError('diagnostic tool imported by runtime: '+name)
    evo=read(EVOLUTION);indexed={r['path']:r for r in evo['changes']}
    required={n for n in parent['files'].keys()|actual.keys() if parent['files'].get(n)!=actual.get(n)}-{EVOLUTION}
    if len(indexed)!=len(evo['changes']) or set(indexed)!=required or evo['parent_review_sha256']!=parent['sha256']:
        raise ValueError('diagnostics evolution coverage mismatch')
    for name,row in indexed.items():
        if row.get('before_sha256')!=parent['files'].get(name) or row.get('after_sha256')!=actual.get(name) or not row.get('reason'):
            raise ValueError('diagnostics evolution hash mismatch: '+name)
    return dict(passed=True,files=len(actual),manifest_sha256=data['sha256'],algorithm_changed=False,
        all_parent_evidence_immutable=True,physics_navigation_reward_and_learning_bytes_unchanged=True,
        original_d2_v1_v2_bytes_unchanged=True,source_admission_preserved=True,
        scope='Passive OLD_ZERO recording, historical reproduction, independent posthoc analysis, and verification metadata')
