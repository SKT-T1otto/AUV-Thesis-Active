"""Declared research priors and finite budgets; numbers are not tuned results."""
import copy
import math

SCHEMA = 'ch3.coupled_search.core.v1'
CORE_DEFAULT = dict(
    schema=SCHEMA, enabled=False, mode='coupled_search', event_capacity=1024, audit_capacity=1024,
    seed=827291, maps=3, particles_per_map=32, horizon=16, execution_scenarios=3,
    raw_routes=12, representatives=4, path_queries=72, proposal_regions=24,
    planning_seconds=20.0, inference_seconds=3.0, history_steps=512,
    geometry_attempts=128, replenish_attempts=2, recovery_cooldown=10,
    geometry=dict(prior_id='public_scan_conditioned_aabb_completion_v1',
                  unknown_boxes=2, max_hit_clusters=8, size_min=0.6, size_max=3.0,
                  cluster_radius=1.2, scan_tolerance=0.25),
    target_prior=dict(prior_id='declared_uniform_position_isotropic_velocity_v1',
                      speed_min=0.2, speed_max=0.8, stationary_mass=0.2),
    beta=0.0, rho=0.25, lambda_cost=0.02, cost_scale=120.0, compression_epsilon=0.02,
    tie_tolerance=0.001, execution_history=8, execution_error_clip=0.5,
    review_samples=64,
)


def validate_core_config(value):
    c = copy.deepcopy(value)
    if not isinstance(c, dict) or set(c) != set(CORE_DEFAULT) or c.get('schema') != SCHEMA:
        raise ValueError('complete coupled-search core configuration required')
    if c['mode'] != 'coupled_search' or type(c['enabled']) is not bool:
        raise ValueError('invalid coupled-search mode or enabled flag')
    limits = dict(event_capacity=(2,8192), audit_capacity=(2,8192), seed=(0,2**32-1),
                  maps=(1,8), particles_per_map=(2,256), horizon=(1,64), execution_scenarios=(1,8),
                  raw_routes=(1,24), representatives=(1,6), path_queries=(1,256), proposal_regions=(1,64),
                  history_steps=(1,2048), geometry_attempts=(1,1024), replenish_attempts=(0,4),
                  recovery_cooldown=(1,100), execution_history=(1,32), review_samples=(1,512))
    for key,(lo,hi) in limits.items():
        if type(c[key]) is not int or not lo <= c[key] <= hi:
            raise ValueError('invalid coupled-search budget: '+key)
    if c['representatives'] > c['raw_routes']:
        raise ValueError('representatives cannot exceed the raw route budget')
    for key in ('planning_seconds','inference_seconds','beta','rho','lambda_cost','cost_scale',
                'compression_epsilon','tie_tolerance','execution_error_clip'):
        if isinstance(c[key], bool) or not isinstance(c[key], (float,int)) or not math.isfinite(c[key]) or c[key] < 0:
            raise ValueError('invalid coupled-search numeric parameter: '+key)
    if min(c[k] for k in ('planning_seconds','inference_seconds','cost_scale','execution_error_clip')) <= 0:
        raise ValueError('positive compute, scale and error bounds required')
    if c['rho'] > 1:
        raise ValueError('risk mixture coefficient must be at most one')
    for key in ('geometry','target_prior'):
        if not isinstance(c[key],dict) or set(c[key]) != set(CORE_DEFAULT[key]):
            raise ValueError('explicit frozen '+key+' required')
        if c[key]['prior_id'] != CORE_DEFAULT[key]['prior_id']:
            raise ValueError('unsupported prior model')
        for name,v in c[key].items():
            if name != 'prior_id' and (isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or v < 0):
                raise ValueError('invalid prior parameter')
    g,p=c['geometry'],c['target_prior']
    if (type(g['unknown_boxes']) is not int or g['unknown_boxes'] > 8
            or type(g['max_hit_clusters']) is not int or not 1 <= g['max_hit_clusters'] <= 32
            or not 0 < g['size_min'] <= g['size_max'] or g['cluster_radius'] <= 0 or g['scan_tolerance'] <= 0
            or not 0 <= p['speed_min'] <= p['speed_max'] or not 0 <= p['stationary_mass'] <= 1):
        raise ValueError('invalid geometry or target prior bounds')
    return c
