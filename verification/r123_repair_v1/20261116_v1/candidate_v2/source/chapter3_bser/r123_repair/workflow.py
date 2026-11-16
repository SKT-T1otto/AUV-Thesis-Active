"""Rule comparisons only. No learner, checkpoint or training entry point."""
import time
import numpy as np
from chapter3_bser.learning.reference import ZeroResidualPolicy
from chapter3_bser.r123.runtime import PathOptionRuntime
from chapter3_bser.r123.workflow import metric as parent_metric
from chapter3_bser.path_option.metrics import aggregate
from chapter3_bser.path_option.workflow import journal
from chapter3_bser.r123_repair.runtime import RuleRuntime,VARIANTS
from chapter3_bser.r123_repair.source import identity,write,sha,digest


def metric(raw,maximum):
    result=parent_metric(raw,maximum);po=raw['path_option'];events=po['lifecycle_events']
    search=[r for r in po['trace'] if r['agent']<3 and r['phase']==0]
    phases={e['task_id']:e['phase'] for e in events if e['event']=='TASK_ACCEPT'}
    result.update(search_wait_steps=sum(r['reason']=='WAIT_TASK' for r in search),
        search_agent_steps=len(search),connector_steps=sum(r['reason']=='CONNECTOR' for r in search),
        root_track_steps=sum(r['reason']=='TRACK_ROOT' for r in search),
        active_recovery_enters=sum(e['event']=='RECOVERY_ENTER' and phases.get(e.get('task_id'))!=3 for e in events),
        frozen_recovery_enters=sum(e['event']=='RECOVERY_ENTER' and phases.get(e.get('task_id'))==3 for e in events),
        release_requests=sum(e['event']=='D2_RELEASE_REQUEST' for e in events),
        rejected_release_requests=sum(e['event']=='D2_RELEASE_REQUEST' and not e['accepted'] for e in events))
    return result


def one_episode(config,scene,opts,variant,*,seed,episode_id,step_hook=None,release_probe=False):
    if variant not in VARIANTS:raise ValueError('unknown rule variant')
    kwargs=dict(seed=seed,episode_id=episode_id,options=opts,arm='rule')
    runtime=PathOptionRuntime(config,scene,**kwargs) if variant=='legacy' else RuleRuntime(config,scene,variant=variant,**kwargs)
    start=time.monotonic();steps=0
    try:
        if release_probe:
            if variant=='legacy':raise ValueError('release probe is for repaired runtimes')
            runtime.observe();runtime.lifecycle.release(0,runtime.step,'VALIDATION_RELEASE_PROBE')
            runtime._release_requests.add(0);runtime._request_refresh=True;runtime._cached=None
        while not runtime.terminal:
            runtime.advance(ZeroResidualPolicy(),explore=False);steps+=1
            if step_hook is not None:step_hook(steps)
            if steps%25==0:print(f'[r123 repair rule {variant}] case={episode_id} step={steps}/{config["max_steps"]} updates=0',flush=True)
        raw=runtime.summary();m=metric(raw,config['max_steps']);m['wall_seconds']=time.monotonic()-start
        return raw,m
    finally:runtime.close()


def evaluate(config,scenes,opts,variant,out,*,seed=982001):
    out.mkdir(parents=True,exist_ok=False);rows=[]
    write(out/'request.json',dict(source=identity(),variant=variant,seed=seed,episodes=len(scenes),
        maximum_physical_steps=len(scenes)*config['max_steps'],training=False,checkpoint_load=False))
    for n,scene in enumerate(scenes):
        raw,m=one_episode(config,scene,opts,variant,seed=seed+n,episode_id=n)
        journal(out/'episodes.jsonl',raw);journal(out/'metrics.jsonl',m);rows.append(m)
        print(f'[r123 repair {variant}] completed={n+1}/{len(scenes)} success={m["success"]} collision={m["collision"]} found={m["found"]}',flush=True)
    result=aggregate(rows);write(out/'summary.json',result)
    write(out/'completed.json',dict(source=identity(),variant=variant,seed=seed,config_hash=digest(config),
        options_hash=digest(opts),scenario_ids=[s['scenario_id'] for s in scenes],
        summary_sha256=sha(out/'summary.json'),episodes_sha256=sha(out/'episodes.jsonl'),
        metrics_sha256=sha(out/'metrics.jsonl'),training_allowed=False,performance_accepted=False))
    return result
