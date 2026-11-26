"""Passive sweep measurements in native physical intervals (zero-based decisions)."""
import numpy as np

from chapter3_bser.experiments.coupled_search_v1.numerics import swept_distances
from core.env.target_motion import EPS


def detection_radii(specs, tolerance):
    """The native detector reads a float32 torch range before adding its Python tolerance."""
    return np.array([float(np.float32(s['sensor_range']))+tolerance for s in specs[:3]])


def swept_cells(centers, start, end, radius, checked):
    if not checked:
        return np.empty(0, dtype=int)
    return np.flatnonzero(swept_distances(start, end, centers, centers) <= radius)


def first_steps(centers, positions, valid, radius):
    """Return [sample, cell] first valid interval, zero for no hit in the horizon."""
    p = np.asarray(positions)
    first = np.zeros((len(p), len(centers)), dtype=np.uint8)
    for k in range(1, p.shape[1]):
        distances = swept_distances(p[:, k-1, None], p[:, k, None], centers, centers)
        hit = (distances <= radius) & (k <= np.asarray(valid)[:, None])
        first[(first == 0) & hit] = k
    return first


def audited_finder(start, end, target0, target1, radii, checked):
    """Independent scalar closest approach. Ties use the original agent index."""
    hits = []
    if checked:
        for i, radius in enumerate(radii):
            before = np.asarray(start[i]) - target0
            after = np.asarray(end[i]) - target1
            relative = after - before
            speed2 = float(relative @ relative)
            phase = 0. if speed2 <= EPS else np.clip(-(before @ relative)/speed2, 0., 1.)
            distance = float(np.linalg.norm(before + phase*relative))
            if distance <= radius:
                hits.append((distance, i))
    return min(hits)[1] if hits else None


def command_key(agent):
    return (agent['task_id'], tuple(agent['root']), tuple(agent['endpoint']), agent['route_id'],
            agent['mode'], agent['hold'], agent['reachable'], agent['released'])


def observed_window(rows, start, agent, horizon, n_cells):
    """Known first hits survive right censoring, unseen cells then remain unknown."""
    reference = command_key(rows[start]['agents'][agent])
    first = np.zeros(n_cells, np.uint8)
    length = 0
    reason = 'complete'
    for k in range(horizon):
        if start+k == len(rows):
            reason = 'external_cap' if not rows[-1]['terminal'] else rows[-1]['outcome']
            break
        row = rows[start+k]
        if not row['checked']:
            reason = 'frozen_after_found' if row['found_before'] else row['outcome']
            break
        if command_key(row['agents'][agent]) != reference:
            reason = 'command_changed'
            break
        length += 1
        cells = np.array(row['hits'][agent], dtype=int)
        first[cells[first[cells] == 0]] = k+1
        if length < horizon and (row['new_finder'] is not None or row['terminal']):
            reason = 'discovered' if row['new_finder'] is not None else row['outcome']
            break
    return first, length, reason
