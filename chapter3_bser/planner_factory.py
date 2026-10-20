"""Original D2 or explicitly enabled coupled-search research assembly."""
from chapter3_bser.d2 import assembly as d2


def prepare(owner, config, scenario, *, map_research=None):
    d2.prepare(owner, config, scenario)
    owner.map_research_config = None
    owner.map_research_session = None
    if map_research is not None:
        from chapter3_bser.experiments.coupled_search_v1.config import validate_research_config
        extension = validate_research_config(map_research)
        if extension['enabled']:
            owner.map_research_config = extension


def before_reset(owner, base, episode_id):
    if owner.map_research_config is not None:
        from chapter3_bser.experiments.coupled_search_v1.assembly import before_reset as hook
        hook(owner, base, episode_id)


def consume(owner):
    if owner.map_research_session is not None:
        from chapter3_bser.experiments.coupled_search_v1.assembly import consume as hook
        hook(owner)


def installed(owner):
    if owner.map_research_session is not None:
        from chapter3_bser.experiments.coupled_search_v1.assembly import installed as hook
        hook(owner)


def build_controller(owner, parent):
    if owner.map_research_config is not None and owner.map_research_config['mode'] == 'coupled_search':
        from chapter3_bser.experiments.coupled_search_v1.assembly import build_controller as build
        return build(owner, parent)
    return d2.build_controller(owner, parent)


def build_bridge(owner):
    return d2.build_bridge(owner)


def diagnostics(owner):
    result = d2.diagnostics(owner)
    if owner.map_research_session is not None:
        result['public_evidence'] = owner.map_research_session.summary()
    return result
