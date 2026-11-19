"""Manual Linux regression 0003-0005, same task-sync control, read-only diagnostics."""
import argparse
import json
import platform
import time
import unittest
from pathlib import Path
from contextlib import redirect_stdout,redirect_stderr
from unittest.mock import patch
from project_paths import ROOT
from chapter3_bser.path_option.source import sha,digest
from chapter3_bser.r123_task_sync.source import identity as control_identity,parent_identity
from tools.r123_task_sync_experiment import read,rows,write,journal,require,fresh,task_metrics

MANIFEST=ROOT/'provenance/r123_task_sync_regression_manifest_v1.json'
INDICES=(3,4,5)


def identity():
    m=read(MANIFEST)
    require(digest({k:v for k,v in m.items() if k!='sha256'})==m['sha256'],'invalid regression manifest')
    require(m['control_source']==control_identity(),'control implementation changed')
    for n,h in m['files'].items():require(sha(ROOT/n)==h,'regression source changed: '+n)
    return dict(schema='auv.r123.task_sync.regression_tool.v1',sha256=m['sha256'],control_source=m['control_source'])


def validation(root):
    root=Path(root);v=read(root/'validation/validation.json')
    require(v['platform']=='Linux' and v['source']==control_identity() and v['passed'] and v['native_smoke_passed'],'same control source Linux validation required')
    require(v['environment_initializations']==4 and v['physical_steps']==128 and v['native_training_updates']==0 and not v['checkpoint_restore'] and v['disabled_matches_parent'],'invalid prior control validation')
    require(v['plan_sha256']==sha(root/'plan/plan.json'),'validated plan changed')
    for n,h in v['artifacts'].items():require(sha(root/'validation'/n)==h,'validation artifact changed: '+n)
    return v


def prepare(old,validated,out):
    old,validated,out=(Path(p).resolve() for p in (old,validated,out))
    require(not any(p==out or p in out.parents or out in p.parents for p in (old,validated)),'output overlaps historical inputs')
    validation(validated);source=identity();done=read(old/'both/completed.json');plan=read(old/'plan/plan.json')
    require(done['source']==plan['source']==parent_identity() and done['variant']=='both' and done['seed']==982001,'wrong original both baseline')
    for n,ext in [('episodes','.jsonl'),('metrics','.jsonl'),('summary','.json')]:require(sha(old/'both'/(n+ext))==done[n+'_sha256'],'baseline result changed')
    for n,h in plan['inputs'].items():require(sha(old/'plan'/n)==h,'original plan changed')
    scenes=read(old/'plan/development.json')['scenarios'][3:6];raw=rows(old/'both/episodes.jsonl')[3:6];metrics=rows(old/'both/metrics.jsonl')[3:6]
    require(len(scenes)==len(raw)==len(metrics)==3,'three regression scenes required')
    require([s['scenario_id'] for s in scenes]==[f'po_eval_992001_{i:04d}' for i in INDICES],'wrong regression scene IDs')
    require(all(e['success'] and not e['task_network_trained'] for e in raw),'expected original successful rule episodes')
    config=read(old/'plan/config.json');require(config['max_steps']==400,'400-step cap required')
    out=fresh(out)
    data={'config.json':config,'options.json':read(old/'plan/options.json'),'scenes.json':scenes,
        'baseline.json':dict(source=done['source'],episodes=raw,metrics=metrics,completion=done)}
    for n,v in data.items():write(out/n,v)
    write(out/'plan.json',dict(tool=source,control_source=control_identity(),indices=list(INDICES),seed=982001,maximum_new_physical_steps=1200,
        validation_origin=str(validated),validation_sha256=sha(validated/'validation/validation.json'),baseline_reused=True,training=False,checkpoint_load=False,
        inputs={n:sha(out/n) for n in data}))
    print(json.dumps(dict(prepared=True,indices=list(INDICES),new_environment_steps=0,output=str(out))))


def load_plan(root):
    root=Path(root);p=read(root/'plan.json');require(p['tool']==identity() and p['control_source']==control_identity(),'plan/source mismatch')
    require(p['indices']==list(INDICES) and p['maximum_new_physical_steps']==1200 and p['seed']==982001,'wrong regression budget or indices')
    for n,h in p['inputs'].items():require(sha(root/n)==h,'plan input changed: '+n)
    validation(p['validation_origin']);require(sha(Path(p['validation_origin'])/'validation/validation.json')==p['validation_sha256'],'validation receipt changed')
    c,o,s=read(root/'config.json'),read(root/'options.json'),read(root/'scenes.json')
    require(c['max_steps']==400 and len(s)==3 and [v['scenario_id'] for v in s]==[f'po_eval_992001_{i:04d}' for i in INDICES],'scene/step cap changed')
    return p,c,o,s


def check(out):
    import checks.test_r123_regression_diagnostics as tests
    out=fresh(out);source=identity()
    with (out/'synthetic.log').open('w',encoding='utf-8') as f:
        with patch('core.env.mission_env.MissionCoreEnv.__init__',side_effect=AssertionError('no environment in observer checks')),redirect_stdout(f),redirect_stderr(f):
            result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(tests))
    passed=result.wasSuccessful() and not result.skipped
    write(out/'checks.json',dict(tool=source,passed=passed,tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        environment_initializations=0,physical_steps=0,training=False,artifacts={'synthetic.log':sha(out/'synthetic.log')}))
    require(passed,'observer checks failed');print(json.dumps(dict(passed=True,tests=result.testsRun,physical_steps=0)))


def run_one(c,s,o,index,*,ledger=None):
    import torch
    from chapter3_bser.r123_task_sync.diagnostics import ObservedRuntime
    from chapter3_bser.learning.reference import ZeroResidualPolicy
    from chapter3_bser.r123_tracking.workflow import metric
    torch.set_num_threads(1)
    if ledger is not None:ledger['environment_initializations']+=1
    r=ObservedRuntime(c,s,seed=982001+index,episode_id=index,options=o,arm='rule',tracking_arm='both',task_sync=True)
    start=time.monotonic()
    try:
        while not r.terminal:
            r.advance(ZeroResidualPolicy(),explore=False);require(r.step<=c['max_steps'],'step cap exceeded')
            if r.step%25==0:print(f'[sync regression] scene={index:04d} step={r.step}/{c["max_steps"]} training=0',flush=True)
        raw=r.summary();m=metric(raw,c['max_steps']);m.update(task_metrics(raw))
        m.update(wall_seconds=time.monotonic()-start,receipt_violations=raw['task_sync']['receipt_violations'],route_queries=raw['task_sync']['route_queries'],
            failure_diagnostic_records=len(raw['failure_diagnostics']['records']),emergency_steps=sum(v['emergency'] for v in raw['failure_diagnostics']['records']))
        require(m['receipt_violations']==0 and m['active_recovery_enters']==0,'unexpected receipt violation or active recovery')
        return raw,m
    finally:
        if ledger is not None:ledger['physical_steps']+=r.step
        r.close()


def evaluate(root,checks,out):
    require(platform.system()=='Linux','complete regression runs are manual Linux only')
    p,c,o,scenes=load_plan(root);v=read(Path(checks)/'checks.json')
    require(v['tool']==identity() and v['passed'] and v['physical_steps']==0,'current observer checks required')
    for n,h in v['artifacts'].items():require(sha(Path(checks)/n)==h,'observer check artifact changed')
    out=fresh(out);source=identity();metrics=[]
    write(out/'request.json',dict(tool=source,plan_sha256=sha(Path(root)/'plan.json'),indices=list(INDICES),maximum_new_physical_steps=1200,training=False,checkpoint_load=False,diagnostics_enabled=True))
    for i,s in zip(INDICES,scenes):
        raw,m=run_one(c,s,o,i);journal(out/'episodes.jsonl',raw);journal(out/'metrics.jsonl',m);metrics.append(m)
        print(f'[sync regression] completed={len(metrics)}/3 scene={i:04d} success={m["success"]} found={m["found"]} collision={m["collision"]}',flush=True)
    require(identity()==source,'source changed during regression')
    write(out/'summary.json',dict(episodes=3,physical_steps=sum(m['steps'] for m in metrics),success=sum(m['success'] for m in metrics),collision=sum(m['collision'] for m in metrics)))
    write(out/'completed.json',dict(tool=source,control_source=control_identity(),plan_sha256=sha(Path(root)/'plan.json'),episodes=3,training=False,checkpoint_load=False,
        artifacts={f.name:sha(f) for f in out.iterdir() if f.is_file()}))


def report(root,results,out):
    p,c,o,scenes=load_plan(root);results=Path(results);done=read(results/'completed.json')
    require(done['tool']==identity() and done['plan_sha256']==sha(Path(root)/'plan.json') and done['episodes']==3,'result identity/count mismatch')
    for n,h in done['artifacts'].items():require(sha(results/n)==h,'result artifact changed: '+n)
    b=read(Path(root)/'baseline.json');new=rows(results/'metrics.jsonl');require(len(new)==3,'three completed metrics required')
    cases=[];keys=('success','found','collision','steps','search_slow_distant_steps','search_wait_steps','search_brake_steps','capture_contact_steps','capture_hold_max')
    for old,raw,now in zip(b['metrics'],b['episodes'],new):
        require(old['scenario_id']==raw['scenario_id']==now['scenario_id'],'paired scene order mismatch')
        normalized=dict(old,**task_metrics(raw))
        cases.append(dict(scenario_id=now['scenario_id'],baseline=normalized,regression=now,paired_delta={k:now[k]-normalized[k] for k in keys},
            task_delta={k:now[k]-normalized[k] for k in task_metrics(raw)}))
    out=fresh(out);write(out/'report.json',dict(tool=identity(),baseline_reused=True,cases=cases,performance_accepted=False,
        notice='Baseline task_accepts normalized to searchers. Observer adds logging cost; wall time is not a controlled speed benchmark. No automatic next experiments.'))
    print(json.dumps(dict(report=str(out/'report.json'),completed_episodes=3)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','check','evaluate','report'))
    p.add_argument('--root',type=Path);p.add_argument('--baseline-root',type=Path);p.add_argument('--validated-run',type=Path);p.add_argument('--checks',type=Path);p.add_argument('--results',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true');a=p.parse_args()
    if not a.execute:print(json.dumps(dict(executed=False,command=a.command)));return
    if a.command=='prepare':require(a.baseline_root is not None and a.validated_run is not None,'--baseline-root and --validated-run required');prepare(a.baseline_root,a.validated_run,a.output)
    elif a.command=='check':check(a.output)
    else:
        require(a.root is not None,'--root required')
        if a.command=='evaluate':require(a.checks is not None,'--checks required');evaluate(a.root,a.checks,a.output)
        else:require(a.results is not None,'--results required');report(a.root,a.results,a.output)


if __name__=='__main__':main()
