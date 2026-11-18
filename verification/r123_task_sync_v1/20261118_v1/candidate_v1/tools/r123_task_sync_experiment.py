"""Bounded task-sync comparison: rule only, immutable imported baseline, no weights."""
import argparse
import copy
import importlib
import json
from pathlib import Path
import platform
import time
import unittest
from contextlib import redirect_stdout,redirect_stderr
from unittest.mock import patch
from project_paths import ROOT
from chapter3_bser.path_option.source import sha,digest
from chapter3_bser.r123_task_sync.source import identity,parent_identity


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_bytes((json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def rows(p):return [json.loads(v) for v in Path(p).read_text(encoding='utf-8').splitlines() if v]
def journal(p,x):
    with Path(p).open('a',encoding='utf-8') as f:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
def fresh(p):p=Path(p);p.mkdir(parents=True,exist_ok=False);return p
def require(ok,message):
    if not ok:raise ValueError(message)


def prepare(old,out):
    old=Path(old).resolve();out=Path(out).resolve()
    require(out!=old and old not in out.parents and out not in old.parents,'plan output overlaps baseline inputs')
    source=identity();done=read(old/'both/completed.json');plan=read(old/'plan/plan.json')
    require(done['source']==plan['source']==parent_identity(),'wrong tracking baseline source')
    require(done['variant']=='both' and done['seed']==982001,'expected original both baseline/seeds')
    for n in ('episodes','metrics','summary'):
        ext='.json' if n=='summary' else '.jsonl'
        require(sha(old/'both'/(n+ext))==done[n+'_sha256'],'baseline artifact changed: '+n)
    for n,h in plan['inputs'].items():require(sha(old/'plan'/n)==h,'baseline plan changed: '+n)
    scenes=read(old/'plan/development.json')['scenarios'][:3]
    require([s['scenario_id'] for s in scenes]==[f'po_eval_992001_{i:04d}' for i in range(3)],'wrong scenes')
    config=read(old/'plan/config.json');require(config['max_steps']==400,'400 step contract required')
    raw=rows(old/'both/episodes.jsonl')[:3];metrics=rows(old/'both/metrics.jsonl')[:3]
    require([e['scenario_id'] for e in raw]==[s['scenario_id'] for s in scenes],'raw scene order mismatch')
    require(all(not e['task_network_trained'] for e in raw),'trained baseline forbidden')
    out=fresh(out)
    inputs={'config.json':config,'options.json':read(old/'plan/options.json'),'scenes.json':scenes,
            'baseline.json':dict(source=done['source'],episodes=raw,metrics=metrics,original_completion=done)}
    for n,x in inputs.items():write(out/n,x)
    write(out/'plan.json',dict(source=source,baseline_source=done['source'],seed=982001,scene_ids=[s['scenario_id'] for s in scenes],
        maximum_new_evaluation_steps=1200,baseline_reused=True,training=False,checkpoint_load=False,inputs={n:sha(out/n) for n in inputs}))
    print(json.dumps(dict(prepared=True,root=str(out),new_evaluation_cap=1200,baseline_reused=True)))


def load_plan(root):
    root=Path(root);plan=read(root/'plan.json');require(plan['source']==identity(),'plan source mismatch')
    for n,h in plan['inputs'].items():require(sha(root/n)==h,'plan input changed: '+n)
    return plan,read(root/'config.json'),read(root/'options.json'),read(root/'scenes.json')


def run_one(config,scene,opts,n,*,enabled=True,parent=False,ledger=None):
    import torch
    from chapter3_bser.learning.reference import ZeroResidualPolicy
    from chapter3_bser.r123_task_sync.runtime import RuleRuntime
    from chapter3_bser.r123_tracking.runtime import RuleRuntime as Tracking
    from chapter3_bser.r123_tracking.workflow import metric
    torch.set_num_threads(1)
    if ledger is not None:ledger['environment_initializations']+=1
    runtime=(Tracking if parent else RuleRuntime)(config,scene,seed=982001+n,episode_id=n,options=opts,arm='rule',tracking_arm='both',**({} if parent else {'task_sync':enabled}))
    start=time.monotonic();last=0
    try:
        while not runtime.terminal:
            runtime.advance(ZeroResidualPolicy(),explore=False)
            require(runtime.step<=config['max_steps'],'physical step cap exceeded')
            if runtime.step%25==0:print(f'[task sync] scene={n:04d} step={runtime.step}/{config["max_steps"]} sync={enabled and not parent} training=0',flush=True)
        raw=runtime.summary();m=metric(raw,config['max_steps'])
        events=raw['path_option']['lifecycle_events'];sync=raw.get('task_sync',{})
        m.update(task_completions=sum(e['event']=='TASK_RELEASE' and e['reason']=='TASK_COMPLETE' and e['agent']<3 for e in events),
            receipt_violations=sync.get('receipt_violations',0),sync_scoring_calls=sync.get('scoring_calls',0),
            sync_scoring_seconds=sync.get('scoring_seconds',0.),route_queries=sync.get('route_queries'),wall_seconds=time.monotonic()-start)
        require(m['receipt_violations']==0,'scoring receipt violation')
        return raw,m
    finally:
        if ledger is not None:ledger['physical_steps']+=runtime.step
        runtime.close()


def validate(root,out,*,native=False):
    plan,config,opts,scenes=load_plan(root);out=fresh(out)
    result=dict(source=identity(),platform=platform.system(),passed=False,native_smoke_passed=False,
        environment_initializations=0,physical_steps=0,native_training_updates=0,checkpoint_restore=False)
    try:
        modules=('test_r123_task_sync','test_r123_tracking','test_r123_governed','test_r123_repair')
        suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(importlib.import_module('checks.'+n)) for n in modules)
        with (out/'synthetic.log').open('w',encoding='utf-8') as f:
            with patch('core.env.mission_env.MissionCoreEnv.__init__',side_effect=AssertionError('native env forbidden in synthetic checks')),redirect_stdout(f),redirect_stderr(f):
                tests=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
        result['synthetic']=dict(tests=tests.testsRun,failures=len(tests.failures),errors=len(tests.errors))
        require(tests.wasSuccessful() and not tests.skipped,'synthetic checks failed')
        if native:
            c=copy.deepcopy(config);c['max_steps']=32
            cases=((0,True,False),(0,False,False),(0,False,True),(2,False,True))
            traces=[]
            for j,(n,parent,enabled) in enumerate(cases):
                s=copy.deepcopy(scenes[n]);s['max_steps']=32
                write(out/f'initialization_{j}.json',dict(scene=n,parent=parent,sync=enabled,max_steps=32,training=False))
                raw,m=run_one(c,s,opts,n,parent=parent,enabled=enabled,ledger=result)
                write(out/f'episode_{j}.json',raw);write(out/f'metric_{j}.json',m)
                traces.append(raw['path_option']['trace'])
                require(m['active_recovery_enters']==0,'active recovery enabled')
                require(not m['collision'],'native collision')
            require(traces[0]==traces[1],'disabled sync changed inherited physical trace')
            result['disabled_matches_parent']=True;result['native_smoke_passed']=True
        require(identity()==result['source'],'source changed during checks');result['passed']=True
    except Exception:
        import traceback
        result['traceback']=traceback.format_exc()
    result['artifacts']={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    write(out/'validation.json',result);print(json.dumps(result,ensure_ascii=False))
    require(result['passed'],'validation failed; see validation.json')


def evaluate(root,validation,out):
    require(platform.system()=='Linux','full evaluation is manual Linux only')
    plan,config,opts,scenes=load_plan(root);v=read(validation)
    require(v['source']==identity() and v['platform']=='Linux' and v['passed'] and v['native_smoke_passed'],'current Linux validation required')
    require(v['environment_initializations']==4 and v['physical_steps']<=128 and v['native_training_updates']==0,'invalid validation budget')
    for n,h in v['artifacts'].items():require(sha(Path(validation).parent/n)==h,'validation artifact changed')
    out=fresh(out);source=identity()
    write(out/'request.json',dict(source=source,plan_sha256=sha(Path(root)/'plan.json'),training=False,checkpoint_load=False,maximum_physical_steps=1200))
    result=[]
    for n,s in enumerate(scenes):
        raw,m=run_one(config,s,opts,n)
        journal(out/'episodes.jsonl',raw);journal(out/'metrics.jsonl',m);result.append(m)
        print(f'[task sync] completed={n+1}/3 success={m["success"]} found={m["found"]} collision={m["collision"]}',flush=True)
    require(source==identity(),'source changed')
    write(out/'summary.json',dict(episodes=3,physical_steps=sum(m['steps'] for m in result),success=sum(m['success'] for m in result),found=sum(m['found'] for m in result)))
    write(out/'completed.json',dict(source=source,plan_sha256=sha(Path(root)/'plan.json'),episodes=3,training=False,checkpoint_load=False,
        artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()}))


def report(root,results,out):
    plan,config,opts,scenes=load_plan(root);done=read(Path(results)/'completed.json')
    require(done['source']==identity() and done['plan_sha256']==sha(Path(root)/'plan.json'),'result source/plan mismatch')
    for n,h in done['artifacts'].items():require(sha(Path(results)/n)==h,'result artifact changed: '+n)
    baseline=read(Path(root)/'baseline.json');new=rows(Path(results)/'metrics.jsonl')
    from chapter3_bser.r123_tracking.workflow import metric
    cases=[]
    keys=('success','found','collision','search_slow_distant_steps','search_brake_steps','search_wait_steps','connector_steps','steps')
    for old,now,raw in zip(baseline['metrics'],new,baseline['episodes']):
        require(old['scenario_id']==now['scenario_id']==raw['scenario_id'],'paired scene mismatch')
        old_extra=sum(e['event']=='TASK_RELEASE' and e['reason']=='TASK_COMPLETE' and e['agent']<3 for e in raw['path_option']['lifecycle_events'])
        cases.append(dict(scenario_id=now['scenario_id'],baseline=old,sync=now,paired_delta={k:now[k]-old[k] for k in keys},task_completions_baseline=old_extra,
            task_completions_delta=now['task_completions']-old_extra))
    out=fresh(out);write(out/'report.json',dict(source=identity(),baseline_reused=True,cases=cases,performance_accepted=False,
        timing_notice='Separate runs; wall time is diagnostic, not a controlled speed benchmark. Baseline route-query counts are unavailable.'))
    print(json.dumps(dict(report=str(out/'report.json'),episodes=3,baseline_reused=True)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('prepare','validate','evaluate','report'))
    p.add_argument('--root',type=Path);p.add_argument('--baseline-root',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--validation',type=Path);p.add_argument('--results',type=Path);p.add_argument('--native-smoke',action='store_true');p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not a.execute:print(json.dumps(dict(executed=False,command=a.command)));return
    if a.command=='prepare':require(a.baseline_root is not None,'--baseline-root required');prepare(a.baseline_root,a.output)
    else:
        require(a.root is not None,'--root required')
        if a.command=='validate':validate(a.root,a.output,native=a.native_smoke)
        elif a.command=='evaluate':require(a.validation is not None,'--validation required');evaluate(a.root,a.validation,a.output)
        else:require(a.results is not None,'--results required');report(a.root,a.results,a.output)


if __name__=='__main__':main()
