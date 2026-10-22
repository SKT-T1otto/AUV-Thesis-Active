"""Strict extension configuration, separate from repair-v2 learning identity."""
import copy

SCHEMA = 'ch3.coupled_search.interface.v1'
DEFAULT = dict(schema=SCHEMA, enabled=False, mode='observe_only',
               event_capacity=1024, audit_capacity=1024)


def validate_research_config(value=None):
    result = copy.deepcopy(DEFAULT if value is None else value)
    if isinstance(result, dict) and result.get('schema') in ('ch3.coupled_search.core.v2','ch3.coupled_search.core.v3'):
        from .core_config import validate_core_config
        return validate_core_config(result)
    if not isinstance(result, dict) or set(result) != set(DEFAULT):
        raise ValueError('research interface requires exactly the declared fields')
    if result['schema'] != SCHEMA or type(result['enabled']) is not bool:
        raise ValueError('invalid research interface schema or enabled flag')
    if result['mode'] != 'observe_only':
        raise ValueError('stage 1 supports observe_only; coupled inference and allocation are not implemented')
    for key in ('event_capacity', 'audit_capacity'):
        if type(result[key]) is not int or not 2 <= result[key] <= 8192:
            raise ValueError(f'{key} must be in [2,8192]')
    return result


def search_planner_id(config):
    if config is None or config.get('mode')!='coupled_search':return 'd2_v1'
    return 'coupled_search_bounded_v3' if config.get('schema')=='ch3.coupled_search.core.v3' else 'coupled_search_v2'
