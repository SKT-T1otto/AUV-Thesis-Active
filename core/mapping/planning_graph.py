from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from typing import Iterable
from typing import Tuple
import numpy as np
_TOPOLOGY_CACHE = {}
_ORACLE_ADJACENCY_CACHE = {}

def _readonly(value, *, dtype=None, shape=None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    locked = np.frombuffer(array.tobytes(order='C'), dtype=array.dtype)
    locked = locked.reshape(array.shape if shape is None else shape)
    locked.setflags(write=False)
    return locked

@dataclass(frozen=True)
class PlanningEdgeView:
    destination: int
    planning_cost: float
    physical_travel_time: float

@dataclass(frozen=True)
class PlanningConnectorView:
    cell_index: int
    component_id: int
    planning_cost: float
    physical_travel_time: float

@dataclass(frozen=True)
class EndpointConnectorSet:
    endpoint_id: str
    role: str
    point: Tuple[float, float, float]
    connectors: Tuple[PlanningConnectorView, ...]
    point_valid: bool = True

@dataclass(frozen=True)
class PlanningGraphView:
    shape: Tuple[int, int, int]
    cell_centers: np.ndarray
    valid_mask: np.ndarray
    component_labels: np.ndarray
    searcher_adjacency: Tuple[Tuple[PlanningEdgeView, ...], ...]
    executor_adjacency: Tuple[Tuple[PlanningEdgeView, ...], ...]
    endpoint_connectors: Tuple[EndpointConnectorSet, ...]
    searcher_horizontal_time_per_unit: float
    searcher_vertical_time_per_unit: float
    executor_horizontal_time_per_unit: float
    executor_vertical_time_per_unit: float
    map_revision: int
    knowledge_mode: str
    graph_sha256: str
    searcher_horizontal_speed_divisor: float = 1.0
    executor_horizontal_speed_divisor: float = 1.0
    searcher_vertical_cost_lookup: Tuple[Tuple[float, float], ...] = ()
    executor_vertical_cost_lookup: Tuple[Tuple[float, float], ...] = ()

    def adjacency_for_role(self, role: str) -> Tuple[Tuple[PlanningEdgeView, ...], ...]:
        return self.executor_adjacency if str(role).lower().startswith('exec') else self.searcher_adjacency

    def time_rates_for_role(self, role: str) -> Tuple[float, float]:
        if str(role).lower().startswith('exec'):
            return (self.executor_horizontal_time_per_unit, self.executor_vertical_time_per_unit)
        return (self.searcher_horizontal_time_per_unit, self.searcher_vertical_time_per_unit)

def _flat(cell: Iterable[int], shape: Tuple[int, int, int]) -> int:
    return int(np.ravel_multi_index(tuple((int(value) for value in cell)), shape))

def _diagnostic_snapshot(planner) -> dict:
    return {name: value for (name, value) in vars(planner).items() if name.startswith('last_') or name.endswith('_count')}

def _topology(planner, shape, valid):
    online = hasattr(planner, 'occupancy_probability')
    identity = 'online' if online else str(getattr(planner, 'obstacle_layout_hash', 'static'))
    key = (type(planner).__qualname__, shape, hashlib.sha256(valid.tobytes(order='C')).hexdigest(), identity)
    cached = _TOPOLOGY_CACHE.get(key)
    if cached is not None:
        return cached
    rows = [[] for _ in range(int(np.prod(shape)))]
    (nx, ny, nz) = shape
    for source in np.flatnonzero(valid):
        cell = tuple((int(value) for value in np.unravel_index(int(source), shape)))
        for offset in planner._neighbor_offsets():
            neighbor = tuple((cell[index] + offset[index] for index in range(3)))
            if not (0 <= neighbor[0] < nx and 0 <= neighbor[1] < ny and (0 <= neighbor[2] < nz)):
                continue
            destination = _flat(neighbor, shape)
            if valid[destination] and planner._edge_is_valid(cell, neighbor):
                rows[int(source)].append(destination)
    topology = tuple((tuple(row) for row in rows))
    _TOPOLOGY_CACHE[key] = topology
    return topology

def _component_labels_from_topology(topology, valid):
    labels = np.full(valid.shape, -1, dtype=np.int64)
    component = 0
    for start in np.flatnonzero(valid):
        if labels[start] >= 0:
            continue
        labels[start] = component
        queue = [int(start)]
        cursor = 0
        while cursor < len(queue):
            source = queue[cursor]
            cursor += 1
            for destination in topology[source]:
                if labels[destination] < 0:
                    labels[destination] = component
                    queue.append(destination)
        component += 1
    return labels

def _build_role_adjacency(planner, role, topology, centers, rates):
    (horizontal_rate, vertical_rate) = rates[role]
    occupancy = None
    if hasattr(planner, 'occupancy_probability'):
        occupancy = np.asarray(planner.occupancy_probability.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
    if occupancy is None:
        cache_key = (id(topology), str(role))
        cached = _ORACLE_ADJACENCY_CACHE.get(cache_key)
        if cached is not None:
            return cached
        output = []
        shape = tuple((int(value) for value in planner.grid_size))
        for (source, destinations) in enumerate(topology):
            left = tuple((int(value) for value in np.unravel_index(source, shape)))
            edges = []
            for destination in destinations:
                right = tuple((int(value) for value in np.unravel_index(destination, shape)))
                value = float(planner._edge_time(left, right, role))
                edges.append(PlanningEdgeView(int(destination), value, value))
            output.append(tuple(edges))
        result = tuple(output)
        _ORACLE_ADJACENCY_CACHE[cache_key] = result
        return result
    output = []
    for (source, destinations) in enumerate(topology):
        if not destinations:
            output.append(())
            continue
        destination_array = np.asarray(destinations, dtype=np.int64)
        delta = np.abs(centers[destination_array] - centers[source])
        physical = np.linalg.norm(delta[:, :2], axis=1) * horizontal_rate + delta[:, 2] * vertical_rate
        probability = 0.5 * (occupancy[source] + occupancy[destination_array])
        entropy = np.zeros(probability.shape, dtype=np.float64)
        interior = (probability > 0.0) & (probability < 1.0)
        p = probability[interior]
        entropy[interior] = -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / np.log(2.0)
        planning = physical + float(planner.occupancy_unknown_cost_weight) * entropy + float(planner.occupancy_risk_cost_weight) * probability * physical
        output.append(tuple((PlanningEdgeView(int(destination), float(plan), float(travel)) for (destination, plan, travel) in zip(destination_array, planning, physical))))
    return tuple(output)

def _build_endpoint(planner, endpoint_id, point, role, labels, shape, centers, valid, rates):
    point_tuple = tuple((float(value) for value in np.asarray(point, dtype=np.float64).reshape(3)))
    if not bool(planner._point_is_valid(point_tuple)):
        return EndpointConnectorSet(str(endpoint_id), str(role), point_tuple, (), False)
    point_array = np.asarray(point_tuple, dtype=np.float64)
    delta = centers - point_array
    (horizontal_rate, vertical_rate) = rates['executor' if str(role).lower().startswith('exec') else 'searcher']
    costs = np.linalg.norm(delta[:, :2], axis=1) * horizontal_rate + np.abs(delta[:, 2]) * vertical_rate
    valid_indices = np.flatnonzero(valid)
    order = valid_indices[np.lexsort((valid_indices, costs[valid_indices]))]
    best_by_component = {}
    component_count = len(set((int(value) for value in labels[valid_indices])))
    for index in order:
        if not planner._segment_is_free_np(point_array, centers[int(index)], getattr(planner, 'planner_obstacle_clearance', 0.0)):
            continue
        component = int(labels[int(index)])
        best_by_component.setdefault(component, (float(costs[int(index)]), int(index)))
        if len(best_by_component) == component_count:
            break
    connectors = []
    for (component, (cost, index)) in sorted(best_by_component.items(), key=lambda item: (item[1][0], item[0], item[1][1])):
        connectors.append(PlanningConnectorView(cell_index=index, component_id=int(component), planning_cost=float(cost), physical_travel_time=float(cost)))
    return EndpointConnectorSet(str(endpoint_id), str(role), point_tuple, tuple(connectors), True)

def _graph_hash(payload) -> str:
    digest = hashlib.sha256()
    scalar = {'shape': payload['shape'], 'map_revision': payload['map_revision'], 'knowledge_mode': payload['knowledge_mode'], 'rates': payload['rates'], 'heuristic': payload['heuristic']}
    digest.update(json.dumps(scalar, sort_keys=True, separators=(',', ':')).encode('utf-8'))
    for array in (payload['centers'], payload['valid'], payload['labels']):
        digest.update(np.asarray(array).tobytes(order='C'))
    for role in ('searcher', 'executor'):
        for (source, edges) in enumerate(payload[role]):
            for edge in edges:
                digest.update(f'{role}|{source}|{edge.destination}|{edge.planning_cost:.17g}|{edge.physical_travel_time:.17g}\n'.encode())
    for endpoint in payload['endpoints']:
        digest.update(f'{endpoint.endpoint_id}|{endpoint.role}|{endpoint.point!r}|{endpoint.point_valid}\n'.encode())
        for connector in endpoint.connectors:
            digest.update(f'{connector.cell_index}|{connector.component_id}|{connector.planning_cost:.17g}|{connector.physical_travel_time:.17g}\n'.encode())
    return digest.hexdigest()

def build_planning_graph(planner, agents, knowledge_mode: str) -> PlanningGraphView:
    """Build a graph view and leave planner cache/diagnostics byte-equivalent."""
    original_cache = planner._geodesic_cache
    diagnostics = _diagnostic_snapshot(planner)
    revisions = {name: getattr(planner, name) for name in ('grid_revision', 'map_revision') if hasattr(planner, name)}
    planner._geodesic_cache = {}
    try:
        shape = tuple((int(value) for value in planner.grid_size))
        centers = np.asarray(planner.flat_xyz_centers.detach().cpu().numpy(), dtype=np.float64).reshape(-1, 3)
        valid = np.asarray(planner.valid_mask.detach().cpu().numpy(), dtype=np.bool_).reshape(-1)
        topology = _topology(planner, shape, valid)
        labels = _component_labels_from_topology(topology, valid)
        rates = {'searcher': (float(planner._continuous_edge_time_np((0, 0, 0), (1, 0, 0), 'searcher')), float(planner._continuous_edge_time_np((0, 0, 0), (0, 0, 1), 'searcher'))), 'executor': (float(planner._continuous_edge_time_np((0, 0, 0), (1, 0, 0), 'executor')), float(planner._continuous_edge_time_np((0, 0, 0), (0, 0, 1), 'executor')))}
        z_points = planner.xyz_centers[0, 0, :, :]
        heuristic = {}
        for role in ('searcher', 'executor'):
            lookup = {0.0: 0.0}
            for left in z_points:
                for right in z_points:
                    delta = abs(float((right[2] - left[2]).item()))
                    lookup.setdefault(delta, float(planner._continuous_edge_time(left, right, role)))
            heuristic[role] = {'horizontal_divisor': 1.0 / rates[role][0], 'vertical_lookup': tuple(sorted(lookup.items()))}
        searcher = _build_role_adjacency(planner, 'searcher', topology, centers, rates)
        executor = _build_role_adjacency(planner, 'executor', topology, centers, rates)
        endpoints = []
        for agent in agents:
            for connector_role in ('searcher', 'executor'):
                endpoints.append(_build_endpoint(planner, f'agent_{agent.agent_id}', agent.position, connector_role, labels, shape, centers, valid, rates))
            if agent.current_navigation_target is not None:
                for connector_role in ('searcher', 'executor'):
                    endpoints.append(_build_endpoint(planner, f'navigation_target_{agent.agent_id}', agent.current_navigation_target, connector_role, labels, shape, centers, valid, rates))
        revision = int(getattr(planner, 'map_revision', 0))
        payload = {'shape': shape, 'centers': centers, 'valid': valid, 'labels': labels, 'searcher': searcher, 'executor': executor, 'endpoints': tuple(endpoints), 'rates': rates, 'heuristic': heuristic, 'map_revision': revision, 'knowledge_mode': str(knowledge_mode)}
        sha = _graph_hash(payload)
        return PlanningGraphView(shape=shape, cell_centers=_readonly(centers, dtype=np.float64), valid_mask=_readonly(valid, dtype=np.bool_), component_labels=_readonly(labels, dtype=np.int64), searcher_adjacency=searcher, executor_adjacency=executor, endpoint_connectors=tuple(endpoints), searcher_horizontal_time_per_unit=rates['searcher'][0], searcher_vertical_time_per_unit=rates['searcher'][1], executor_horizontal_time_per_unit=rates['executor'][0], executor_vertical_time_per_unit=rates['executor'][1], map_revision=revision, knowledge_mode=str(knowledge_mode), graph_sha256=sha, searcher_horizontal_speed_divisor=heuristic['searcher']['horizontal_divisor'], executor_horizontal_speed_divisor=heuristic['executor']['horizontal_divisor'], searcher_vertical_cost_lookup=heuristic['searcher']['vertical_lookup'], executor_vertical_cost_lookup=heuristic['executor']['vertical_lookup'])
    finally:
        planner._geodesic_cache = original_cache
        for (name, value) in diagnostics.items():
            setattr(planner, name, value)
        for (name, value) in revisions.items():
            if getattr(planner, name) != value:
                setattr(planner, name, value)
