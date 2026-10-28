"""Independent monitoring-only evolution audit against immutable e613ecb."""
import ast
import hashlib
import json
import zipfile
from pathlib import Path
from tools.reach_avoid_geometry_source import inventory as parent_inventory
from tools.map_interface_source import digest

MANIFEST='provenance/reach_avoid_monitor_manifest.json'
EVOLUTION='provenance/reach_avoid_monitor_evolution.json'
PARENT='provenance/reach_avoid_geometry_manifest.json'
BASELINE='provenance/baselines/reach_avoid_geometry_20261028_v2'
PROFILE='active_reach_avoid_monitor_v3'


def inventory(root,*,allow_missing_evolution=False):
    root=Path(root);actual=parent_inventory(root)
    actual[PARENT]=hashlib.sha256((root/PARENT).read_bytes()).hexdigest()
    if not allow_missing_evolution or (root/EVOLUTION).exists():actual[EVOLUTION]=hashlib.sha256((root/EVOLUTION).read_bytes()).hexdigest()
    return actual


class RemoveMetadata(ast.NodeTransformer):
    def visit_If(self,node):
        if any(isinstance(n,ast.Constant) and n.value==MANIFEST for n in ast.walk(node.test)):return None
        return self.generic_visit(node)
    def visit_Dict(self,node):
        pairs=[(k,v) for k,v in zip(node.keys,node.values) if not isinstance(k,ast.Constant) or k.value!=PROFILE]
        node.keys=[k for k,v in pairs];node.values=[v for k,v in pairs];return self.generic_visit(node)
    def visit_Tuple(self,node):
        node.elts=[n for n in node.elts if not isinstance(n,ast.Constant) or n.value!=PROFILE];return self.generic_visit(node)


def workflow_without_observer(text):
    """Remove only exact reviewed calls; a mutation hidden in an if is rejected."""
    exact=["from chapter3_bser.reach_avoid.monitoring import TrainingMonitor",
        "monitor=TrainingMonitor.from_args(args,output,source,config,scenes,options,training_ids,fingerprints) if training else None",
        "if not training and getattr(args,'dev_scenarios',None):raise ValueError('development monitoring is training-only')"]
    for call in ('monitor.start(learner)',"monitor.on_update(metric,learner,0,0,'pretrain')",
                 'monitor.after_pretrain(learner)','monitor.before_episode(learner)',
                 "monitor.on_update(metric,learner,episode+1,steps,'online')",
                 'monitor.on_episode(learner,summary,episode+1,steps,training_ids,fingerprints)',
                 'monitor.finish(learner,training_ids,fingerprints)'):
        exact.append('if monitor:'+call)
    exact.extend(["p.add_argument('--dev-scenarios')","p.add_argument('--monitor-settings',default=str(ROOT/'configs/reach_avoid_monitor.json'))"])
    allowed={ast.dump(ast.parse(s).body[0],include_attributes=False) for s in exact}
    class Remove(ast.NodeTransformer):
        def visit(self,node):
            if isinstance(node,ast.stmt) and ast.dump(node,include_attributes=False) in allowed:return None
            return super().visit(node)
    text=text.replace("'active_reach_avoid_geometry_v2','active_reach_avoid_monitor_v3'","'active_reach_avoid_geometry_v2'")
    text=text.replace("records.append(dict(termination_reason=summary['termination_reason'],success=summary['success']))","records.append(summary)")
    return Remove().visit(ast.parse(text))


def verify(root):
    root=Path(root);read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    sha=lambda n:hashlib.sha256((root/n).read_bytes()).hexdigest()
    actual=inventory(root);data=read(MANIFEST);parent=read(PARENT);base=read(BASELINE+'/baseline.json')
    if data.get('schema')!='auv.active_source.reach_avoid_monitor.v3' or data.get('sha256')!=digest({k:v for k,v in data.items() if k!='sha256'}):raise ValueError('invalid monitoring manifest')
    if actual!=data['files']:raise ValueError('independent source inventory mismatch')
    for name in actual:
        if not name.startswith(('core/','chapter3_bser/')) or not name.endswith('.py'):continue
        for node in ast.walk(ast.parse((root/name).read_text(encoding='utf-8'))):
            imports=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
            if any(n=='tools' or n.startswith('tools.') for n in imports):raise ValueError('tool imported by runtime: '+name)
    if data.get('algorithm_changed') is not False or data['parent_review_sha256']!=parent['sha256'] or data['parent_manifest_sha256']!=sha(PARENT):raise ValueError('immutable geometry parent changed')
    if base['source_identity']['review_sha256']!=parent['sha256'] or sha(BASELINE+'/source_and_docs.zip')!=base['archive_sha256']:raise ValueError('immutable parent archive changed')
    with zipfile.ZipFile(root/BASELINE/'source_and_docs.zip') as archive:
        external=base['external_immutable_files']
        if set(archive.namelist())!=set(base['files'])-set(external):raise ValueError('parent archive inventory changed')
        for name,h in base['files'].items():
            if name in external and sha(name)!=h:raise ValueError('parent evidence changed: '+name)
            if name not in external and hashlib.sha256(archive.read(name)).hexdigest()!=h:raise ValueError('parent archive member changed: '+name)
            if name.startswith('provenance/') and sha(name)!=h:raise ValueError('parent provenance changed: '+name)
        for name,h in parent['files'].items():
            if not name.startswith(('core/','chapter3_bser/')) or actual.get(name)==h:continue
            if name=='chapter3_bser/learning/provenance.py':
                new=RemoveMetadata().visit(ast.parse((root/name).read_text(encoding='utf-8')))
                if ast.dump(new,include_attributes=False)!=ast.dump(ast.parse(archive.read(name).decode()),include_attributes=False):raise ValueError('source admission changed beyond validation metadata')
                continue
            if name=='chapter3_bser/reach_avoid/workflow.py':
                new=workflow_without_observer((root/name).read_text(encoding='utf-8'))
                if ast.dump(new,include_attributes=False)!=ast.dump(ast.parse(archive.read(name).decode()),include_attributes=False):raise ValueError('workflow changed beyond exact observer hooks')
                continue
            if name=='core/env/mission_env.py':raise ValueError('public handoff read-only API changed')
            if name.startswith('core/env/'):raise ValueError('legacy AST changed: '+name)
            if name.endswith('geometry_hypotheses.py'):raise ValueError('protected geometry kernel changed: '+name)
            if name.endswith('bounded_belief.py'):raise ValueError('observer changed algorithm AST: '+name)
            raise ValueError('protected parent runtime changed: '+name)
    additions={n for n in actual.keys()-parent['files'].keys() if n.startswith(('core/','chapter3_bser/'))}
    if additions!={'chapter3_bser/reach_avoid/learning_diagnostics.py','chapter3_bser/reach_avoid/monitoring.py',
                  'chapter3_bser/reach_avoid/monitor_reports.py','chapter3_bser/learning/reach_avoid_monitor_source.py'}:raise ValueError('unexpected monitoring runtime extension')
    evo=read(EVOLUTION);indexed={r['path']:r for r in evo['changes']}
    required={n for n in parent['files'].keys()|actual.keys() if parent['files'].get(n)!=actual.get(n)}-{EVOLUTION}
    if len(indexed)!=len(evo['changes']) or set(indexed)!=required or evo['parent_review_sha256']!=parent['sha256']:raise ValueError('monitoring evolution coverage mismatch')
    for name,row in indexed.items():
        if row.get('before_sha256')!=parent['files'].get(name) or row.get('after_sha256')!=actual.get(name) or not row.get('reason'):raise ValueError('monitoring evolution hash mismatch: '+name)
    return dict(passed=True,files=len(actual),manifest_sha256=data['sha256'],algorithm_changed=False,
        all_parent_evidence_immutable=True,original_D2_B2_B3_safe_rl_unchanged=True,
        learner_controller_reward_dynamics_unchanged=True,scope='Read-only empirical learning monitoring and explicit research launcher')
