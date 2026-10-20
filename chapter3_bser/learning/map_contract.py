"""Explicit native-map experiment identity, detached from B2/B3 algorithms."""
import copy
import re
import time
import numpy as np
from chapter3_bser.learning.checkpoint import digest

SCHEMA='auv.active.map_condition.v1'


def validate_condition(value):
    if not isinstance(value,dict) or set(value)!={'schema','map_id','research_config','protocol_sha256'}:
        raise ValueError('complete map experiment condition required')
    c=copy.deepcopy(value)
    if c['schema']!=SCHEMA or c['map_id'] not in ('OLD','NEW') or not isinstance(c['protocol_sha256'],str) or not re.fullmatch('[0-9a-f]{64}',c['protocol_sha256']):
        raise ValueError('invalid map experiment identity')
    if c['map_id']=='OLD':
        if c['research_config'] is not None:raise ValueError('OLD must use original default-off D2')
    else:
        from chapter3_bser.experiments.coupled_search_v1.core_config import validate_core_config
        c['research_config']=validate_core_config(c['research_config'])
        if c['research_config']['enabled'] is not True:raise ValueError('NEW must explicitly enable coupled search')
    return c


def identity(config):
    if 'map_experiment' not in config:return None
    c=validate_condition(config['map_experiment'])
    return dict(schema=SCHEMA,map_id=c['map_id'],search_planner_id='d2_v1' if c['map_id']=='OLD' else 'coupled_search_v1',
                condition_sha256=digest(c),research_config_sha256=digest(c['research_config']),protocol_sha256=c['protocol_sha256'])


def research_config(config):
    return None if 'map_experiment' not in config else validate_condition(config['map_experiment'])['research_config']


def metadata(config):
    value=identity(config)
    return {} if value is None else dict(map_condition=value,train_map=value['map_id'],eval_map=value['map_id'])


def validate_checkpoint_map(config,payload):
    expected=identity(config)
    if expected is None:
        if payload.get('map_condition') is not None:raise ValueError('unexpected checkpoint map identity')
    elif (payload.get('map_condition')!=expected or payload.get('train_map')!=expected['map_id']
            or payload.get('eval_map')!=expected['map_id']):
        raise ValueError('checkpoint native map identity mismatch; cross-map evaluation is not this protocol')
    return expected


def advance(runtime,model,*,explore):
    if 'map_experiment' not in runtime.config:return runtime.advance(model,explore=explore)
    started=time.perf_counter()
    try:return runtime.advance(model,explore=explore)
    finally:
        if not hasattr(runtime,'_map_step_seconds'):runtime._map_step_seconds=[]
        runtime._map_step_seconds.append(time.perf_counter()-started)


def metrics(runtime):
    if 'map_experiment' not in runtime.config:return {}
    condition=identity(runtime.config);session=runtime.map_research_session
    audit=getattr(runtime.controller.allocator,'audit',[])
    proposals=[r for r in audit if r.get('kind')=='coupled_proposal']
    elapsed=[r['elapsed_seconds'] for r in proposals]
    steps=getattr(runtime,'_map_step_seconds',[])
    eps=[r['epsilon'] for p in proposals for r in p['compression'].values()]
    return dict(map_condition=condition,train_map=condition['map_id'],eval_map=condition['map_id'],map_metrics=dict(
        core_proposals=len(proposals),d2_fallbacks=0 if session is None else session.counts.get('d2_fallbacks',0),
        path_queries=sum(p['path_queries'] for p in proposals),compression_max_epsilon=max(eps,default=None),
        core_planning_seconds=sum(elapsed),core_planning_p95=None if not elapsed else float(np.percentile(elapsed,95)),
        runtime_step_seconds=sum(steps),runtime_step_p50=None if not steps else float(np.percentile(steps,50)),
        runtime_step_p95=None if not steps else float(np.percentile(steps,95)),
        final_filter_valid=None if session is None else session.belief.valid,
        planning_timing_scope='core successful proposals only; runtime timing includes environment and actor'))
