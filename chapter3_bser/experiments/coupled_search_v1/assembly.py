"""Public evidence capture and optional coupled-search controller composition."""
from .audit import PublicEvidenceSession


def before_reset(owner, base, episode_id):
    config = owner.map_research_config
    base.enable_public_event_capture(episode_id, capacity=config['event_capacity'])
    session_type = PublicEvidenceSession
    if config['mode'] == 'coupled_search':
        from .session import CoupledSearchSession
        session_type = CoupledSearchSession
    owner.map_research_session = session_type(config, base.get_public_vehicle_specs(),
                                             base.get_public_control_contract())


def consume(owner):
    session = owner.map_research_session
    if not session.valid:
        if session.config['mode']=='coupled_search':raise RuntimeError('coupled_search_public_evidence_invalid:'+str(session.error))
        return
    try:
        events = owner.env.read_public_events(session.adapter.cursor)
    except ValueError as error:
        session.valid, session.error = False, str(error)
        session.counts['event_history_failures'] += 1
        if session.config['mode']=='coupled_search':raise RuntimeError('coupled_search_public_history_lost') from error
        return
    session.consume(events)
    if session.config['mode']=='coupled_search' and not session.valid:
        raise RuntimeError('coupled_search_public_evidence_invalid:'+str(session.error))


def installed(owner):
    session = owner.map_research_session
    if session.config['mode'] == 'coupled_search':
        indices = {i: owner.bridge.path_tracker.snapshot(i).next_index for i in range(3)}
        session.installed(owner.guidance, indices)
        session.activity.install(owner.guidance,owner.controller.current_allocation)
    else:
        session.installed(owner.guidance)


def build_controller(owner, parent):
    from .allocator import CoupledSearchAllocator
    from .controller import CoupledController, CoupledPolicy
    from chapter3_bser.d2.controller import D2SearchController
    allocator=CoupledSearchAllocator(owner,parent.allocator.config,'D2',owner.env.get_agent_state().positions[3])
    inner=CoupledController(parent.config,allocator)
    inner.prrac_runtime_contract=parent.prrac_runtime_contract
    wrapped=D2SearchController(inner,owner)
    inner.policy=CoupledPolicy(inner.config,allocator)
    return wrapped
