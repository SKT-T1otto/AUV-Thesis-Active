"""Carry completed route vertices across a public query, with fresh geometry."""
import numpy as np


def carry_progress(old, fresh, position, geometry, radius):
    """Skip only a completed prefix, up to the unchanged old tracking vertex.

    Never edit paths or map cells, carry an unreachable route, skip a new detour,
    or connect through geometry that is no longer verified. The caller must
    establish that both routes belong to the same live task and phase.
    """
    before = fresh.index
    if not old.reachable or not fresh.reachable or not len(old.points) or not len(fresh.points):
        return dict(advanced=False, reason='unreachable_or_empty', skipped=0)
    fresh.advance(position, radius)
    start = fresh.index
    if old.index <= 0 or old.index >= len(old.points):
        return dict(advanced=False, reason='no_completed_prefix', skipped=0)
    anchor = old.points[old.index]
    completed = old.points[:old.index]
    matches = np.flatnonzero(np.linalg.norm(fresh.points - anchor, axis=1) <= 1e-4)
    for end in matches:
        if end <= start:
            continue
        prefix = fresh.points[start:end]
        if not all(np.any(np.linalg.norm(completed - p, axis=1) <= 1e-4) for p in prefix):
            continue
        if not geometry.segment(np.asarray(position), fresh.points[end]):
            return dict(advanced=False, reason='fresh_geometry_veto', skipped=0)
        fresh.index = int(end)
        return dict(advanced=True, reason='completed_prefix_connected', skipped=int(end-start),
                    previous_query_index=int(before), tracking=fresh.points[end].tolist())
    return dict(advanced=False, reason='no_matching_completed_prefix', skipped=0)


def tracking_velocity(packet, agent, route, endpoint, enabled):
    """Only a final moving-target point has the target's velocity."""
    if packet.phases[agent] != 2:
        return np.zeros(3)
    if not enabled:
        return packet.target_velocity.copy()
    terminal = route.reachable and len(route.points) > 0 and route.index == len(route.points)-1
    if terminal and np.linalg.norm(route.tracking(packet.positions[agent])-endpoint) <= 1e-4:
        return packet.target_velocity.copy()
    return np.zeros(3)
