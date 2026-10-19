from __future__ import annotations
import numpy as np
from chapter3_bser.types import SearchCandidate
from core.mapping.planning_state import PlanningStateView

def _point_segment_distance(points: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    segment = right - left
    denominator = float(np.dot(segment, segment))
    if denominator <= 1e-18:
        return np.linalg.norm(points - left, axis=1)
    projection = np.clip((points - left) @ segment / denominator, 0.0, 1.0)
    nearest = left[None, :] + projection[:, None] * segment[None, :]
    return np.linalg.norm(points - nearest, axis=1)

def detection_probability(candidate: SearchCandidate, state: PlanningStateView, config: dict) -> np.ndarray:
    agent = state.agents[candidate.agent_id]
    parameters = config['detection_model']
    p_max = float(parameters['p_max_by_role'][agent.role]) * float(parameters.get('p_scale', 1.0))
    if agent.sensor_radius is None or agent.sensor_radius <= 0.0:
        raise ValueError('search candidate requires a positive public sensor radius')
    sigma = float(parameters['sigma_sensor_radius_multiplier']) * agent.sensor_radius
    centers = np.asarray(state.grid.cell_centers)
    path = np.asarray(candidate.path_points)
    if path.shape[0] == 1:
        distance = np.linalg.norm(centers - path[0], axis=1)
    else:
        distance = np.full(centers.shape[0], np.inf, dtype=np.float64)
        for index in range(path.shape[0] - 1):
            distance = np.minimum(distance, _point_segment_distance(centers, path[index], path[index + 1]))
    q_free = 1.0 - np.asarray(state.occupancy.occupancy_probability, dtype=np.float64)
    probability = p_max * np.exp(-distance ** 2 / (2.0 * sigma ** 2)) * q_free
    probability = np.clip(probability, 0.0, 1.0)
    if not np.all(np.isfinite(probability)):
        raise ValueError('detection kernel produced a non-finite value')
    probability.setflags(write=False)
    return probability
