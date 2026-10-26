"""Versioned old-map executor navigation and phase-specific residual policy."""
import copy
SCHEMA='ch3.executor_recovery.v2'
OPTIONS=dict(schema=SCHEMA,enabled=True,motion_steps=8,braking_steps=16,
    route_retry_steps=5,tracking_distance=.9,recovery_candidates=7,
    require_known_free=True,executor_search_residual='disabled')


def validate_options(value):
    if not isinstance(value,dict):raise ValueError('executor safety options must be an object')
    if value.get('schema')!=SCHEMA:
        from chapter3_bser.safety.executor_standby import validate_options as v1
        return v1(value)
    c=copy.deepcopy(value)
    if (set(c)!=set(OPTIONS) or type(c['enabled']) is not bool
        or any(type(c[k]) is not int or c[k]!=OPTIONS[k] for k in ('motion_steps','braking_steps','recovery_candidates'))
        or type(c['route_retry_steps']) is not int or not 1<=c['route_retry_steps']<=20
        or type(c['tracking_distance']) not in (int,float) or not .2<=c['tracking_distance']<=1.2
        or c['require_known_free'] is not True or c['executor_search_residual'] not in ('disabled','original')):
        raise ValueError('invalid bounded executor recovery v2 options')
    return c
