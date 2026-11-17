"""Rule comparisons only. No learner, checkpoint or training entry point."""
import time
import numpy as np
from chapter3_bser.learning.reference import ZeroResidualPolicy
from chapter3_bser.r123.runtime import PathOptionRuntime
from chapter3_bser.r123.workflow import metric as parent_metric
from chapter3_bser.path_option.workflow import journal
from chapter3_bser.r123_tracking.runtime import RuleRuntime,ARMS
VARIANTS=tuple(ARMS)
from chapter3_bser.r123_tracking.source import identity,write,sha,digest


def aggregate(rows):
    if not rows:raise ValueError('no completed episodes')
    result=dict(episodes=len(rows),present_counts={})
    for key in rows[0]:
        if key=='scenario_id':continue
        values=[r[key] for r in rows if r[key] is not None]
        result[key]=float(np.mean(values)) if values else None
        result['present_counts'][key]=len(values)
    return result


def metric(raw,maximum):
    from chapter3_bser.r123_governed.workflow import metric as inherited_metric
    result=inherited_metric(raw,maximum)
    trace=raw['path_option']['trace']
    capture=[r for r in trace if r['agent']==3 and r['phase']==2]
    fixed=[r for r in capture if np.linalg.norm(np.asarray(r['event_target'])-r['endpoint'])>1e-4]
    result.update(route_continuity_advances=sum(e['advanced'] for e in raw['path_option'].get('route_continuity_events',[])),
        capture_fixed_waypoint_steps=len(fixed),
        capture_static_feedforward_steps=sum(np.linalg.norm(r.get('action_diagnostics',{}).get('tracking_velocity',[0.,0.,0.]))>1e-9 for r in fixed),
        capture_brake_steps=sum(r['decision']['action']==8 for r in capture),
        executor_min_distance=raw.get('executor_min_distance_to_target'),
        found_step=raw.get('found_step'),success_step=raw.get('success_step'))
    # Old governed traces did not log tracking_velocity: phase-2 code always
    # supplied target velocity. Report this counter as unknown, never as zero.
    if fixed and any('tracking_velocity' not in r.get('action_diagnostics',{}) for r in fixed):
        result['capture_static_feedforward_steps']=None
    return {k:v.item() if isinstance(v,np.generic) else v for k,v in result.items()}


def one_episode(config,scene,opts,variant,*,seed,episode_id,step_hook=None,release_probe=False):
    if variant not in VARIANTS:raise ValueError('unknown rule variant')
    kwargs=dict(seed=seed,episode_id=episode_id,options=opts,arm='rule',tracking_arm=variant)
    runtime=RuleRuntime(config,scene,**kwargs)
    start=time.monotonic();steps=0
    try:
        if release_probe:
            if variant=='legacy':raise ValueError('release probe is for repaired runtimes')
            runtime.observe();runtime.lifecycle.release(0,runtime.step,'VALIDATION_RELEASE_PROBE')
            runtime._release_requests.add(0);runtime._request_refresh=True;runtime._cached=None
        while not runtime.terminal:
            runtime.advance(ZeroResidualPolicy(),explore=False);steps+=1
            if step_hook is not None:step_hook(steps)
            if steps%25==0:print(f'[r123 tracking rule {variant}] case={episode_id} step={steps}/{config["max_steps"]} updates=0',flush=True)
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
        print(f'[r123 tracking {variant}] completed={n+1}/{len(scenes)} success={m["success"]} collision={m["collision"]} found={m["found"]}',flush=True)
    result=aggregate(rows);write(out/'summary.json',result)
    write(out/'completed.json',dict(source=identity(),variant=variant,switches=dict(zip(('path','feedforward'),ARMS[variant])),seed=seed,config_hash=digest(config),
        options_hash=digest(opts),scenario_ids=[s['scenario_id'] for s in scenes],
        summary_sha256=sha(out/'summary.json'),episodes_sha256=sha(out/'episodes.jsonl'),
        metrics_sha256=sha(out/'metrics.jsonl'),training_allowed=False,performance_accepted=False))
    return result
