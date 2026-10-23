"""V3 allocation constraints from public geometry; unavailable agents keep hold."""
from chapter3_bser.safety.planning_refresh import start_status


def feasible_groups(state,routes,expected,assigned):
    """Omit only unassigned agents whose public start has no safe connector.

    A candidate shortage at a valid start is still an error. Existing semantic
    assignments are never discarded here; partial recovery keeps its original
    atomic rejection when it cannot assign every affected agent.
    """
    available={route.agent_id for route in routes};missing=set(expected)-available
    held={}
    for agent_id in sorted(missing):
        if agent_id in assigned:raise ValueError('missing_public_safe_route')
        status=start_status(state,agent_id)
        if status not in ('invalid_start','no_start_connector'):
            raise ValueError('missing_public_safe_route')
        # The start status comes from the same filtered public graph that route
        # queries use. No ground truth or fabricated motion candidate is added.
        held[str(agent_id)]=status
    active=set(expected)-missing
    if not active:raise ValueError('missing_public_safe_route')
    return active,held
