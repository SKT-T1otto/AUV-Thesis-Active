from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from typing import Optional
from typing import Tuple
import numpy as np
import torch
from core.mapping.planning_graph import PlanningGraphView
from core.mapping.planning_graph import build_planning_graph

def _readonly(value, *, dtype=None) -> np.ndarray:
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy().copy()
    else:
        array = np.asarray(value, dtype=dtype).copy()
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    array.setflags(write=False)
    return array

@dataclass(frozen=True)
class GridGeometryView:
    shape: Tuple[int, int, int]
    origin: Tuple[float, float, float]
    spacing: Tuple[float, float, float]
    cell_centers: np.ndarray

@dataclass(frozen=True)
class TargetBeliefView:
    probabilities: np.ndarray
    entropy: float
    peak_probability: float
    peak_index: int
    revision: int
    normalized: bool

@dataclass(frozen=True)
class OccupancyBeliefView:
    occupancy_probability: np.ndarray
    known_mask: np.ndarray
    free_mask: np.ndarray
    occupied_mask: np.ndarray
    unknown_mask: np.ndarray
    revision: int
    knowledge_mode: str

@dataclass(frozen=True)
class PlanningAgentView:
    agent_id: int
    role: str
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    horizontal_speed_limit: float
    vertical_speed_limit: float
    sensor_radius: Optional[float]
    current_navigation_target: Optional[Tuple[float, float, float]]

@dataclass(frozen=True)
class PlanningStateView:
    step: int
    target_found: bool
    mission_complete: bool
    grid: GridGeometryView
    target_belief: TargetBeliefView
    occupancy: OccupancyBeliefView
    agents: Tuple[PlanningAgentView, ...]
    executor_id: int
    searcher_ids: Tuple[int, int, int]
    map_revision: int
    knowledge_mode: str
    planning_graph: PlanningGraphView

def planning_state_sha256(state: PlanningStateView) -> str:
    """Hash algorithm-visible state while deliberately excluding metadata/step."""
    digest = hashlib.sha256()
    scalar = {'target_found': state.target_found, 'mission_complete': state.mission_complete, 'executor_id': state.executor_id, 'searcher_ids': state.searcher_ids, 'map_revision': state.map_revision, 'knowledge_mode': state.knowledge_mode, 'planning_graph_sha256': state.planning_graph.graph_sha256, 'agents': [{'agent_id': agent.agent_id, 'role': agent.role, 'position': agent.position, 'velocity': agent.velocity, 'horizontal_speed_limit': agent.horizontal_speed_limit, 'vertical_speed_limit': agent.vertical_speed_limit, 'sensor_radius': agent.sensor_radius, 'current_navigation_target': agent.current_navigation_target} for agent in state.agents]}
    digest.update(json.dumps(scalar, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8'))
    for array in (state.grid.cell_centers, state.target_belief.probabilities, state.occupancy.occupancy_probability, state.occupancy.known_mask, state.occupancy.free_mask, state.occupancy.occupied_mask, state.occupancy.unknown_mask):
        digest.update(np.asarray(array).tobytes(order='C'))
    return digest.hexdigest()

def _triple(row) -> Tuple[float, float, float]:
    values = np.asarray(row, dtype=np.float64).reshape(3)
    return tuple((float(value) for value in values))

def extract_planning_state(env) -> PlanningStateView:
    """Copy the current planner-visible state without changing the environment.

    Unknown-map profiles expose only the live online occupancy arrays.  A
    planner that has no probabilistic occupancy arrays is the declared oracle
    planner; its *current* validity mask is represented as a deterministic
    occupancy snapshot.
    """
    task = env.get_task_state()
    public_agents = env.get_agent_state()
    identity = env.get_scenario_identity()
    runtime = env.unwrapped
    planner = runtime.map_module
    centers = _readonly(planner.flat_xyz_centers, dtype=np.float64).reshape(-1, 3)
    shape = tuple((int(value) for value in planner.grid_size))
    spacing = (float(planner.cell_dx), float(planner.cell_dy), float(planner.cell_dz))
    origin = tuple((float(centers[0, axis] - 0.5 * spacing[axis]) for axis in range(3)))
    belief = _readonly(planner.belief_map, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(belief)):
        raise ValueError('target belief contains non-finite values')
    if np.any(belief < -1e-12):
        raise ValueError('target belief contains negative values')
    belief_sum = float(np.sum(belief, dtype=np.float64))
    normalized = abs(belief_sum - 1.0) <= 1e-06
    if not normalized:
        raise ValueError(f'target belief sum is {belief_sum}, expected one')
    positive = belief[belief > 0.0]
    entropy = float(-np.sum(positive * np.log(positive), dtype=np.float64))
    peak_index = int(np.argmax(belief))
    knowledge_mode = str(identity['obstacle_knowledge_mode'])
    if hasattr(planner, 'occupancy_probability'):
        occupancy = _readonly(planner.occupancy_probability, dtype=np.float64).reshape(-1)
        free = _readonly(planner.known_free_mask, dtype=np.bool_).reshape(-1)
        occupied = _readonly(planner.known_occupied_mask, dtype=np.bool_).reshape(-1)
        unknown = _readonly(planner.unknown_mask, dtype=np.bool_).reshape(-1)
        known = _readonly(free | occupied, dtype=np.bool_)
        revision = int(planner.map_revision)
    else:
        free = _readonly(planner.valid_mask, dtype=np.bool_).reshape(-1)
        occupied = _readonly(~free, dtype=np.bool_)
        unknown = _readonly(np.zeros(free.shape, dtype=np.bool_))
        known = _readonly(np.ones(free.shape, dtype=np.bool_))
        occupancy = _readonly(occupied.astype(np.float64))
        revision = int(getattr(planner, 'grid_revision', 0))
    if not np.all(np.isfinite(occupancy)) or np.any(occupancy < 0.0) or np.any(occupancy > 1.0):
        raise ValueError('occupancy probability must be finite and in [0, 1]')
    agents = []
    for (agent_id, (role, position, velocity, nav_target, spec)) in enumerate(zip(public_agents.role_order, public_agents.positions, public_agents.velocities, public_agents.navigation_targets, runtime.agent_specs)):
        sensor = float(spec['sensor_range'])
        agents.append(PlanningAgentView(agent_id=agent_id, role=str(role), position=_triple(position), velocity=_triple(velocity), horizontal_speed_limit=float(spec['v_xy_max']), vertical_speed_limit=float(spec['v_z_max']), sensor_radius=None if sensor <= 0.0 else sensor, current_navigation_target=_triple(nav_target)))
    grid = GridGeometryView(shape=shape, origin=origin, spacing=spacing, cell_centers=centers)
    target = TargetBeliefView(probabilities=belief, entropy=entropy, peak_probability=float(belief[peak_index]), peak_index=peak_index, revision=revision, normalized=normalized)
    occupancy_view = OccupancyBeliefView(occupancy_probability=occupancy, known_mask=known, free_mask=free, occupied_mask=occupied, unknown_mask=unknown, revision=revision, knowledge_mode=knowledge_mode)
    planning_graph = build_planning_graph(planner, tuple(agents), knowledge_mode)
    return PlanningStateView(step=int(task.step), target_found=bool(task.target_found), mission_complete=bool(task.mission_complete), grid=grid, target_belief=target, occupancy=occupancy_view, agents=tuple(agents), executor_id=3, searcher_ids=(0, 1, 2), map_revision=revision, knowledge_mode=knowledge_mode, planning_graph=planning_graph)
