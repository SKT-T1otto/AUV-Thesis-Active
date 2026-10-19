from __future__ import annotations
from dataclasses import replace
from collections import OrderedDict
import hashlib
import json
import numpy as np
from core.mapping.planning_state import PlanningStateView
_FILTER_GEOMETRY_CACHE = OrderedDict()
_FILTER_GEOMETRY_CACHE_LIMIT = 16

def _locked(value, dtype=None):
    array = np.asarray(value, dtype=dtype)
    return np.frombuffer(array.tobytes(order='C'), dtype=array.dtype).reshape(array.shape)

def _vector(point):
    try:
        result = np.asarray(tuple(point), dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('point must be a finite three-vector') from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError('point must be a finite three-vector')
    return result

class PublicGeometry:
    """Closed-segment collision checks against one immutable public snapshot."""

    def __init__(self, state: PlanningStateView, clearance: float=0.0):
        self.clearance = float(clearance)
        if not np.isfinite(self.clearance) or self.clearance < 0.0:
            raise ValueError('clearance must be finite and nonnegative')
        shape = tuple(state.grid.shape)
        if len(shape) != 3 or any((int(n) != n or n < 1 for n in shape)):
            raise ValueError('grid shape must contain three positive integers')
        self.lower = _locked(_vector(state.grid.origin))
        spacing = _vector(state.grid.spacing)
        if np.any(spacing < 0.0) or any((spacing[a] == 0.0 and shape[a] != 1 for a in range(3))):
            raise ValueError('zero grid spacing is only valid on a singleton axis')
        self.upper = _locked(self.lower + np.asarray(shape) * spacing)
        centers = np.asarray(state.grid.cell_centers, dtype=np.float64)
        count = int(np.prod(shape))
        if centers.shape != (count, 3) or not np.all(np.isfinite(centers)):
            raise ValueError('public cell centers do not match grid shape')
        expected = np.asarray(list(np.ndindex(shape)), dtype=np.float64)
        expected = self.lower + (expected + 0.5) * spacing
        tolerance = 8.0 * np.finfo(np.float32).eps * max(1.0, float(np.max(np.abs(centers))))
        if not np.allclose(centers, expected, rtol=0.0, atol=tolerance):
            raise ValueError('public cell centers do not match origin and spacing')
        occupied = np.asarray(state.occupancy.occupied_mask, dtype=np.bool_)
        if occupied.shape != (count,):
            raise ValueError('occupied mask does not match grid shape')
        half_extent = 0.5 * spacing + self.clearance
        self.occupied_lower = _locked(centers[occupied] - half_extent)
        self.occupied_upper = _locked(centers[occupied] + half_extent)

    def _segments_free(self, starts, ends):
        """Vectorized slab test; zero-length segments are point tests."""
        starts = np.asarray(starts, dtype=np.float64).reshape(-1, 3)
        ends = np.asarray(ends, dtype=np.float64).reshape(-1, 3)
        if starts.shape != ends.shape:
            raise ValueError('segment endpoint arrays must have matching shapes')
        result = np.all(np.isfinite(starts), axis=1) & np.all(np.isfinite(ends), axis=1) & np.all(starts >= self.lower, axis=1) & np.all(starts <= self.upper, axis=1) & np.all(ends >= self.lower, axis=1) & np.all(ends <= self.upper, axis=1)
        if not len(self.occupied_lower):
            return result
        batch_size = max(1, min(256, 131072 // max(1, len(self.occupied_lower))))
        for offset in range(0, len(starts), batch_size):
            left = starts[offset:offset + batch_size, None, :]
            delta = ends[offset:offset + batch_size, None, :] - left
            parallel = delta == 0.0
            outside = parallel & ((left < self.occupied_lower) | (left > self.occupied_upper))
            near = np.full((len(left), len(self.occupied_lower), 3), -np.inf)
            far = np.full_like(near, np.inf)
            first = np.zeros_like(near)
            second = np.zeros_like(near)
            np.divide(self.occupied_lower - left, delta, out=first, where=~parallel)
            np.divide(self.occupied_upper - left, delta, out=second, where=~parallel)
            np.copyto(near, np.minimum(first, second), where=~parallel)
            np.copyto(far, np.maximum(first, second), where=~parallel)
            entry = np.maximum(0.0, np.max(near, axis=2))
            leave = np.minimum(1.0, np.min(far, axis=2))
            touches = ~np.any(outside, axis=2) & (entry <= leave)
            result[offset:offset + len(left)] &= ~np.any(touches, axis=1)
        return result

    def point_free(self, point) -> bool:
        try:
            value = _vector(point)
        except ValueError:
            return False
        return bool(self._segments_free(value[None, :], value[None, :])[0])

    def segment_free(self, start, end) -> bool:
        try:
            (left, right) = (_vector(start), _vector(end))
        except ValueError:
            return False
        return bool(self._segments_free(left[None, :], right[None, :])[0])

    def path_free(self, points) -> bool:
        try:
            values = np.asarray([_vector(point) for point in points], dtype=np.float64)
        except (TypeError, ValueError):
            return False
        if not len(values):
            return False
        if len(values) == 1:
            return self.point_free(values[0])
        return bool(np.all(self._segments_free(values[:-1], values[1:])))

def _component_labels(valid, searcher, executor):
    """Shared weak components; each role's existing A* still tests its own edges."""
    neighbors = [set() for _ in valid]
    for adjacency in (searcher, executor):
        for (source, row) in enumerate(adjacency):
            for edge in row:
                neighbors[source].add(edge.destination)
                neighbors[edge.destination].add(source)
    labels = np.full(len(valid), -1, dtype=np.int64)
    component = 0
    for start in np.flatnonzero(valid):
        if labels[start] >= 0:
            continue
        labels[start] = component
        queue = [int(start)]
        for source in queue:
            for destination in sorted(neighbors[source]):
                if labels[destination] < 0:
                    labels[destination] = component
                    queue.append(destination)
        component += 1
    return labels

def _graph_hash(graph, clearance):
    digest = hashlib.sha256()
    scalar = {'filter': 'safe_search_public_closed_voxel_v1', 'clearance': clearance}
    for (name, value) in vars(graph).items():
        if name not in ('cell_centers', 'valid_mask', 'component_labels', 'searcher_adjacency', 'executor_adjacency', 'endpoint_connectors', 'graph_sha256'):
            scalar[name] = value
    digest.update(json.dumps(scalar, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    for array in (graph.cell_centers, graph.valid_mask, graph.component_labels):
        digest.update(array.tobytes(order='C'))
    for role in ('searcher', 'executor'):
        for (source, row) in enumerate(graph.adjacency_for_role(role)):
            for edge in row:
                digest.update(f'{role}|{source}|{edge.destination}|{edge.planning_cost:.17g}|{edge.physical_travel_time:.17g}\n'.encode())
    for endpoint in graph.endpoint_connectors:
        digest.update(repr(endpoint).encode())
    return digest.hexdigest()

def filter_planning_state(state: PlanningStateView, clearance: float=0.0) -> PlanningStateView:
    """Remove unsafe geometry, retaining original costs and the original A*.

    This function never invents a connector or snaps a continuous endpoint to a
    grid cell. A legal endpoint whose supplied connectors are all removed stays
    legal but disconnected. Callers must refresh the authoritative public graph
    to obtain a missing current-position endpoint.
    """
    geometry = PublicGeometry(state, clearance)
    graph = state.planning_graph
    centers = np.asarray(graph.cell_centers, dtype=np.float64)
    count = len(centers)
    if graph.shape != state.grid.shape or not np.array_equal(centers, state.grid.cell_centers):
        raise ValueError('planning graph and public grid geometry differ')
    valid = np.asarray(graph.valid_mask, dtype=np.bool_)
    if valid.shape != (count,):
        raise ValueError('graph valid mask does not match cell centers')
    digest = hashlib.sha256()
    for array in (centers, geometry.lower, geometry.upper, geometry.occupied_lower, geometry.occupied_upper):
        digest.update(repr((array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes(order='C'))
    digest.update(repr(geometry.clearance).encode())
    geometry_key = digest.hexdigest()
    cached = _FILTER_GEOMETRY_CACHE.get(geometry_key)
    if cached is None:
        cached = (_locked(geometry._segments_free(centers, centers)), {})
        _FILTER_GEOMETRY_CACHE[geometry_key] = cached
        while len(_FILTER_GEOMETRY_CACHE) > _FILTER_GEOMETRY_CACHE_LIMIT:
            _FILTER_GEOMETRY_CACHE.popitem(last=False)
    else:
        _FILTER_GEOMETRY_CACHE.move_to_end(geometry_key)
    (node_free, edge_free) = cached
    valid = valid & node_free
    pairs = set()
    for adjacency in (graph.searcher_adjacency, graph.executor_adjacency):
        if len(adjacency) != count:
            raise ValueError('graph adjacency does not match cell centers')
        for (source, row) in enumerate(adjacency):
            for edge in row:
                destination = int(edge.destination)
                if destination != edge.destination or not 0 <= destination < count:
                    raise ValueError('graph edge destination is invalid')
                if not np.isfinite(edge.planning_cost) or not np.isfinite(edge.physical_travel_time) or min(edge.planning_cost, edge.physical_travel_time) < 0.0:
                    raise ValueError('graph edge costs must be finite and nonnegative')
                if valid[source] and valid[destination]:
                    pairs.add(tuple(sorted((source, destination))))
    missing = sorted(pairs - edge_free.keys())
    if missing:
        indices = np.asarray(missing, dtype=np.int64)
        free = geometry._segments_free(centers[indices[:, 0]], centers[indices[:, 1]])
        edge_free.update(((pair, bool(accepted)) for (pair, accepted) in zip(missing, free)))
    safe_pairs = {pair for pair in pairs if edge_free[pair]}

    def filtered(adjacency):
        return tuple((tuple((edge for edge in row if tuple(sorted((source, edge.destination))) in safe_pairs)) for (source, row) in enumerate(adjacency)))
    (searcher, executor) = (filtered(graph.searcher_adjacency), filtered(graph.executor_adjacency))
    labels = _component_labels(valid, searcher, executor)
    endpoints = []
    for endpoint in graph.endpoint_connectors:
        point_valid = bool(endpoint.point_valid and geometry.point_free(endpoint.point))
        connectors = []
        for connector in endpoint.connectors:
            index = int(connector.cell_index)
            if index != connector.cell_index or not 0 <= index < count:
                raise ValueError('endpoint connector index is invalid')
            if not np.isfinite(connector.planning_cost) or not np.isfinite(connector.physical_travel_time) or min(connector.planning_cost, connector.physical_travel_time) < 0.0:
                raise ValueError('endpoint costs must be finite and nonnegative')
            if point_valid and valid[index] and geometry.segment_free(endpoint.point, centers[index]):
                connectors.append(replace(connector, component_id=int(labels[index])))
        endpoints.append(replace(endpoint, point_valid=point_valid, connectors=tuple(connectors)))
    result = replace(graph, cell_centers=_locked(centers), valid_mask=_locked(valid), component_labels=_locked(labels), searcher_adjacency=searcher, executor_adjacency=executor, endpoint_connectors=tuple(endpoints))
    result = replace(result, graph_sha256=_graph_hash(result, geometry.clearance))
    return replace(state, planning_graph=result)
