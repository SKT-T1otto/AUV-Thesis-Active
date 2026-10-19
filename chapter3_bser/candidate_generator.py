from __future__ import annotations
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple
import numpy as np
from chapter3_bser.types import CandidateGenerationResult
from chapter3_bser.types import SearchCandidate
from chapter3_bser.types import StandbyCandidate
from core.mapping.planning_state import PlanningStateView
from core.mapping.travel_cost_service import TravelCostService

def _local_peak_indices(values: np.ndarray, shape: Tuple[int, int, int]) -> List[int]:
    cube = values.reshape(shape)
    result = []
    for cell in np.ndindex(shape):
        value = cube[cell]
        neighbours = []
        for offset in np.ndindex((3, 3, 3)):
            delta = tuple((item - 1 for item in offset))
            if delta == (0, 0, 0):
                continue
            other = tuple((cell[axis] + delta[axis] for axis in range(3)))
            if all((0 <= other[axis] < shape[axis] for axis in range(3))):
                neighbours.append(cube[other])
        if not neighbours or value >= max(neighbours):
            result.append(int(np.ravel_multi_index(cell, shape)))
    return result

def _frontier_indices(state: PlanningStateView) -> List[int]:
    unknown = np.asarray(state.occupancy.unknown_mask).reshape(state.grid.shape)
    free = np.asarray(state.occupancy.free_mask).reshape(state.grid.shape)
    frontier = []
    for cell in np.ndindex(state.grid.shape):
        if not unknown[cell]:
            continue
        adjacent_free = False
        for axis in range(3):
            for direction in (-1, 1):
                other = list(cell)
                other[axis] += direction
                if 0 <= other[axis] < state.grid.shape[axis] and free[tuple(other)]:
                    adjacent_free = True
        if adjacent_free:
            frontier.append(int(np.ravel_multi_index(cell, state.grid.shape)))
    return frontier

def _ordered_pool(state: PlanningStateView) -> List[Tuple[int, str]]:
    belief = np.asarray(state.target_belief.probabilities)
    occupied = np.asarray(state.occupancy.occupied_mask)
    free = np.asarray(state.occupancy.free_mask)
    ranking = np.lexsort((np.arange(belief.size), -belief))
    categories = (('belief_local_peak', [index for index in sorted(_local_peak_indices(belief, state.grid.shape), key=lambda item: (-belief[item], item)) if not occupied[index]]), ('high_belief_uncovered', [int(index) for index in ranking if not occupied[index]]), ('occupancy_frontier', [index for index in sorted(_frontier_indices(state), key=lambda item: (-belief[item], item)) if not occupied[index]]), ('known_free_high_belief', [int(index) for index in ranking if free[index] and (not occupied[index])]))
    sources: Dict[int, str] = {}
    depth = 0
    while any((depth < len(indices) for (_, indices) in categories)):
        for (source, indices) in categories:
            if depth < len(indices):
                sources.setdefault(indices[depth], source)
        depth += 1
    return list(sources.items())

def _separated(index: int, selected: Sequence[int], centers: np.ndarray, distance: float) -> bool:
    return all((float(np.linalg.norm(centers[index] - centers[other])) >= distance - 1e-12 for other in selected))

def generate_search_candidates(state: PlanningStateView, service: TravelCostService, *, k_search: int, minimum_separation: float, maximum_travel_time: float) -> Tuple[Tuple[SearchCandidate, ...], int, Tuple[str, ...]]:
    centers = np.asarray(state.grid.cell_centers)
    pool = _ordered_pool(state)
    output: List[SearchCandidate] = []
    unreachable = 0
    reasons: List[str] = []
    for agent_id in state.searcher_ids:
        agent = state.agents[agent_id]
        selected_indices: List[int] = []
        for (index, source) in pool:
            if len(selected_indices) >= int(k_search):
                break
            if not _separated(index, selected_indices, centers, float(minimum_separation)):
                continue
            query = service.query(agent.position, centers[index], agent)
            if not query.reachable:
                unreachable += 1
                continue
            if query.physical_travel_time > float(maximum_travel_time):
                reasons.append(f'agent_{agent_id}:PATH_BUDGET_EXCEEDED:{index}')
                continue
            candidate_id = f's{agent_id}_{len(selected_indices):02d}_{index:04d}'
            output.append(SearchCandidate(agent_id=agent_id, candidate_id=candidate_id, waypoint=tuple((float(value) for value in centers[index])), path_points=query.path_points, path_cell_indices=query.path_cell_indices, path_length=query.path_length, planning_cost=query.planning_cost, physical_travel_time=query.physical_travel_time, source=source))
            selected_indices.append(index)
        if len(selected_indices) < int(k_search):
            reasons.append(f'agent_{agent_id}:ONLY_{len(selected_indices)}_OF_{int(k_search)}')
    output.sort(key=lambda candidate: candidate.key)
    return (tuple(output), unreachable, tuple(reasons))

def generate_standby_candidates(state: PlanningStateView, service: TravelCostService, *, k_standby: int) -> Tuple[Tuple[StandbyCandidate, ...], int, Tuple[str, ...]]:
    executor = state.agents[state.executor_id]
    centers = np.asarray(state.grid.cell_centers)
    belief = np.asarray(state.target_belief.probabilities)
    occupied = np.asarray(state.occupancy.occupied_mask)
    proposals: List[Tuple[Tuple[float, float, float], str]] = [(executor.position, 'current_position')]
    if executor.current_navigation_target is not None:
        proposals.append((executor.current_navigation_target, 'current_standby'))
    valid = np.flatnonzero(~occupied)
    ranking = valid[np.lexsort((valid, -belief[valid]))]
    if ranking.size:
        proposals.append((tuple(centers[int(ranking[0])]), 'belief_peak'))
        weighted = np.sum(centers[valid] * belief[valid, None], axis=0) / max(float(np.sum(belief[valid])), 1e-12)
        nearest = int(valid[np.argmin(np.sum((centers[valid] - weighted) ** 2, axis=1))])
        proposals.append((tuple(centers[nearest]), 'belief_weighted_representative'))
        proposals.extend(((tuple(centers[int(index)]), 'high_probability_region') for index in ranking))
    output: List[StandbyCandidate] = []
    seen = set()
    unreachable = 0
    for (waypoint, source) in proposals:
        if len(output) >= int(k_standby):
            break
        key = tuple((round(float(value), 10) for value in waypoint))
        if key in seen:
            continue
        seen.add(key)
        query = service.query(executor.position, waypoint, executor)
        if not query.reachable:
            unreachable += 1
            continue
        output.append(StandbyCandidate(candidate_id=f'y{len(output):02d}', waypoint=tuple((float(value) for value in waypoint)), path_points=query.path_points, path_cell_indices=query.path_cell_indices, path_length=query.path_length, planning_cost=query.planning_cost, physical_travel_time=query.physical_travel_time, source=source))
    reasons = () if len(output) >= int(k_standby) else (f'standby:ONLY_{len(output)}_OF_{int(k_standby)}',)
    output.sort(key=lambda candidate: candidate.key)
    return (tuple(output), unreachable, reasons)

def generate_candidates(state: PlanningStateView, config: dict, *, k_search=None, k_standby=None) -> CandidateGenerationResult:
    generation = config['candidate_generation']
    search_k = int(generation['k_search_exact'] if k_search is None else k_search)
    standby_k = int(generation['k_standby_exact'] if k_standby is None else k_standby)
    service = TravelCostService(state)
    (search, unreachable_search, search_reasons) = generate_search_candidates(state, service, k_search=search_k, minimum_separation=float(generation['minimum_separation']), maximum_travel_time=float(generation.get('maximum_physical_travel_time', generation.get('maximum_travel_time'))))
    (standby, unreachable_standby, standby_reasons) = generate_standby_candidates(state, service, k_standby=standby_k)
    counts = {agent_id: sum((candidate.agent_id == agent_id for candidate in search)) for agent_id in state.searcher_ids}
    return CandidateGenerationResult(search, standby, counts, unreachable_search, unreachable_standby, search_reasons + standby_reasons)
