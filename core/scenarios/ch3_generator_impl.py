from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import numpy as np
import torch
from core.config.ch3_constants import CH3_UNKNOWN_MAP_V1
from core.config.ch3_constants import DEFAULT_VALIDATION_SCENARIO_COUNT
from core.config.ch3_constants import SCENARIO_SPLITS
from core.config.ch3_constants import UNKNOWN_MAP_PROFILES
from core.config.ch3_constants import UNKNOWN_SMOKE_TRAIN_GENERATOR_SEED
from core.config.ch3_constants import UNKNOWN_SMOKE_VALIDATION_GENERATOR_SEED
from core.config.ch3_constants import UNKNOWN_TRAIN_GENERATOR_SEED
from core.config.ch3_constants import UNKNOWN_VALIDATION_GENERATOR_SEED
from core.env.target_motion import TargetState
from core.env.target_motion import simulate_target_trajectory
from core.mapping.path_planner import ObstacleAwareTaskMapPlanner
SPACE_SIZE = np.asarray([20.0, 20.0, 8.0], dtype=np.float64)
PLANNER_GRID_SIZE = (10, 10, 8)
PLANNER_OBSTACLE_CLEARANCE = 0.4
TARGET_OBSTACLE_CLEARANCE = 0.2
TARGET_CAPTURE_RADIUS = 0.8
PROFILE_CODES = {'S00_STATIC_CLEAR': 's00_static_clear', 'S10_MOVING_CLEAR': 's10_moving_clear', 'S01_STATIC_OBSTACLE': 's01_static_obstacle', 'S11_MOVING_OBSTACLE': 's11_moving_obstacle', 'M00_MOVING_CLEAR': 'm00', 'M10_MOVING_UNKNOWN_SINGLE': 'm10', 'M20_MOVING_UNKNOWN_MULTI': 'm20', 'M90_MOVING_KNOWN_ORACLE': 'm90'}
_BUILD_CACHE = {}
_UNKNOWN_PROFILE_KNOWLEDGE = {'M00_MOVING_CLEAR': 'online_unknown', 'M10_MOVING_UNKNOWN_SINGLE': 'online_unknown', 'M20_MOVING_UNKNOWN_MULTI': 'online_unknown', 'M90_MOVING_KNOWN_ORACLE': 'oracle'}
_UNKNOWN_PROFILE_FAMILY = {'M00_MOVING_CLEAR': 'clear', 'M10_MOVING_UNKNOWN_SINGLE': 'random_single_aabb_v1', 'M20_MOVING_UNKNOWN_MULTI': 'random_multi_aabb_v1', 'M90_MOVING_KNOWN_ORACLE': 'random_multi_aabb_v1'}

def _canonical_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()

def _inside(point, obstacles, clearance=0.0):
    point = np.asarray(point, dtype=np.float64)
    for obstacle in obstacles:
        center = np.asarray(obstacle['center'], dtype=np.float64)
        half = np.asarray(obstacle['size'], dtype=np.float64) / 2.0 + clearance
        if np.all(point >= center - half) and np.all(point <= center + half):
            return True
    return False

def _sample_point(rng, obstacles, margin=1.0, clearance=0.5):
    for _ in range(4096):
        point = np.asarray([rng.uniform(margin, SPACE_SIZE[0] - margin), rng.uniform(margin, SPACE_SIZE[1] - margin), rng.uniform(0.8, 7.2)])
        if not _inside(point, obstacles, clearance):
            return point
    raise RuntimeError('unable to sample a legal mission point')

def _planner(obstacles, agent_positions):
    planner = ObstacleAwareTaskMapPlanner(space_size=SPACE_SIZE, grid_size=PLANNER_GRID_SIZE, z_range=(0.8, 7.2), planner_obstacle_clearance=PLANNER_OBSTACLE_CLEARANCE, device='cpu', dtype=torch.float32)
    planner.reset(agent_positions=torch.as_tensor(agent_positions, dtype=torch.float32), obstacles=obstacles)
    return planner

def _boxes_overlap(left, right, margin=0.35):
    left_center = np.asarray(left['center'], dtype=np.float64)
    left_half = np.asarray(left['size'], dtype=np.float64) / 2.0 + margin
    right_center = np.asarray(right['center'], dtype=np.float64)
    right_half = np.asarray(right['size'], dtype=np.float64) / 2.0 + margin
    return bool(np.all(np.abs(left_center - right_center) <= left_half + right_half))

def _sample_unknown_agents(rng):
    points = []
    for _ in range(4):
        for _ in range(4096):
            point = _sample_point(rng, ())
            if all((np.linalg.norm(point - previous) >= 2.8 for previous in points)):
                points.append(point)
                break
        else:
            raise RuntimeError('unable to sample separated agent positions')
    return np.stack(points)

def _sample_unknown_obstacles(rng, count, protected_points):
    obstacles = []
    protected = [np.asarray(point, dtype=np.float64) for point in protected_points]
    for _ in range(int(count)):
        accepted = None
        for _ in range(4096):
            size = np.asarray([rng.uniform(1.4, 4.0), rng.uniform(1.4, 5.0), rng.uniform(1.2, 4.8)])
            half = size / 2.0
            center = np.asarray([rng.uniform(half[0] + 0.8, SPACE_SIZE[0] - half[0] - 0.8), rng.uniform(half[1] + 0.8, SPACE_SIZE[1] - half[1] - 0.8), rng.uniform(half[2] + 0.5, SPACE_SIZE[2] - half[2] - 0.5)])
            candidate = {'center': center.round(7).tolist(), 'size': size.round(7).tolist()}
            if any((_inside(point, [candidate], clearance=1.0) for point in protected)):
                continue
            if any((_boxes_overlap(candidate, previous) for previous in obstacles)):
                continue
            accepted = candidate
            break
        if accepted is None:
            raise RuntimeError('unable to sample a non-overlapping obstacle')
        obstacles.append(accepted)
    return obstacles

def _unknown_trajectory_record(position, velocity, obstacles):
    state = TargetState(position, velocity, 0, 'constant_velocity_reflect_v1', state_schema='moving_target_state_v1', obstacle_layout_id='custom_aabb_v1' if obstacles else 'none')
    trajectory = simulate_target_trajectory(state, 400, 0.2, SPACE_SIZE, obstacles, clearance=TARGET_OBSTACLE_CLEARANCE, max_reflections=4, max_prediction_steps=400)
    points = np.stack([item.position for item in trajectory])
    if not np.all(np.isfinite(points)):
        raise RuntimeError('target trajectory is non-finite')
    if np.any(points < -1e-08) or np.any(points > SPACE_SIZE + 1e-08):
        raise RuntimeError('target trajectory left the world')
    if any((_inside(point, obstacles, clearance=TARGET_OBSTACLE_CLEARANCE - 1e-08) for point in points)):
        raise RuntimeError('target trajectory entered an expanded obstacle')
    distance = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    if distance <= 10.0:
        raise RuntimeError('moving target trajectory is too short')
    identity = {'dt': 0.2, 'max_steps': 400, 'target_obstacle_clearance': TARGET_OBSTACLE_CLEARANCE, 'motion_mode': 'constant_velocity_reflect_v1', 'obstacle_layout_hash': _canonical_sha(obstacles), 'points': points.round(8).tolist()}
    return (_canonical_sha(identity), distance, points)

def _validate_unknown_truth_reachability(agent_positions, wait_point, target_point, trajectory_points, obstacles):
    """Use obstacle truth only for dataset feasibility construction."""
    planner = _planner(obstacles, agent_positions)
    labels = planner._component_labels()

    def connector_map(point):
        connectors = {}
        for (cost, cell) in planner._connector_candidates(point):
            component = labels.get(cell)
            if component is not None:
                connectors[int(component)] = (float(cost), cell)
        return connectors
    endpoints = [*[np.asarray(point, dtype=np.float64) for point in agent_positions], np.asarray(wait_point, dtype=np.float64), np.asarray(target_point, dtype=np.float64)]
    endpoint_connectors = [connector_map(point) for point in endpoints]
    if any((not item for item in endpoint_connectors)):
        raise RuntimeError('endpoint has no oracle connector')
    common = set.intersection(*[set(item) for item in endpoint_connectors])
    if not common:
        raise RuntimeError('endpoints have no common oracle component')
    component = min(common)
    reference = endpoints[0]
    for point in endpoints:
        path = planner.grid_astar_path(reference, point, role='executor')
        if not path['reachable'] or path.get('resolved_component_id', component) != component or (not path.get('start_connector_valid', True)) or (not path.get('goal_connector_valid', True)) or (not path.get('exact_goal_reachable', True)):
            raise RuntimeError('endpoint is not exactly oracle-reachable')
    max_connector_distance = 0.0
    for point in trajectory_points:
        connectors = connector_map(point)
        if component not in connectors:
            raise RuntimeError('target trajectory enters an oracle-unreachable component')
        (_, cell) = connectors[component]
        center = planner._xyz_centers_np[cell]
        max_connector_distance = max(max_connector_distance, float(np.linalg.norm(point - center)))
        if not planner.segment_is_free(planner.xyz_centers[cell], point):
            raise RuntimeError('target trajectory has a blocked final connector')
    component_id = _canonical_sha({'layout': _canonical_sha(obstacles), 'component': int(component), 'grid_size': PLANNER_GRID_SIZE, 'clearance': PLANNER_OBSTACLE_CLEARANCE})[:16]
    return {'connectivity_component_id': component_id, 'target_trajectory_reachable': True, 'target_trajectory_exact_endpoint_reachable': True, 'target_trajectory_max_connector_distance': float(max_connector_distance), 'target_trajectory_connector_failure_count': 0}

def _unknown_profile_obstacles(profile, single, multi):
    if profile == 'M00_MOVING_CLEAR':
        return []
    if profile == 'M10_MOVING_UNKNOWN_SINGLE':
        return single
    return multi

def _build_unknown_profile_manifests(count, generator_seed, split):
    count = int(count)
    generator_seed = int(generator_seed)
    split = str(split)
    if split not in SCENARIO_SPLITS:
        raise ValueError(f'unsupported split={split!r}; expected {SCENARIO_SPLITS}')
    if count <= 0:
        raise ValueError('scenario count must be positive')
    cache_key = ('unknown', count, generator_seed, split)
    if cache_key in _BUILD_CACHE:
        return deepcopy(_BUILD_CACHE[cache_key])
    rng = np.random.default_rng(generator_seed)
    manifests = {profile: {'protocol': CH3_UNKNOWN_MAP_V1, 'manifest_id': f'{CH3_UNKNOWN_MAP_V1}_{split}_{PROFILE_CODES[profile]}_scenarios_v1', 'scenario_profile': profile, 'scenario_role': split, 'scenario_split': split, 'scenario_schema_version': 1, 'scenario_count': count, 'generator_seed': generator_seed, 'max_steps': 400, 'target_motion_known': True, 'scenarios': []} for profile in UNKNOWN_MAP_PROFILES}
    for index in range(count):
        records = None
        for _attempt in range(96):
            agents = _sample_unknown_agents(rng)
            target = _sample_point(rng, ())
            wait = _sample_point(rng, ())
            direction = rng.normal(size=3)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                continue
            direction /= norm
            velocity = direction * rng.uniform(0.25, 0.55)
            protected = [*agents, target, wait]
            try:
                single = _sample_unknown_obstacles(rng, 1, protected)
                multi = _sample_unknown_obstacles(rng, int(rng.integers(2, 5)), protected)
                profile_records = {}
                for profile in UNKNOWN_MAP_PROFILES:
                    obstacles = _unknown_profile_obstacles(profile, single, multi)
                    (trajectory_sha, trajectory_distance, trajectory_points) = _unknown_trajectory_record(target, velocity, obstacles)
                    reachability = _validate_unknown_truth_reachability(agents, wait, target, trajectory_points, obstacles)
                    profile_records[profile] = {'obstacles': deepcopy(obstacles), 'trajectory_sha': trajectory_sha, 'trajectory_distance': trajectory_distance, 'reachability': reachability}
            except (RuntimeError, ValueError):
                continue
            records = {'agents': agents, 'target': target, 'wait': wait, 'velocity': velocity, 'profiles': profile_records, 'planner_seed': int(rng.integers(1, 2 ** 31 - 1))}
            break
        if records is None:
            raise RuntimeError(f'unable to sample unknown-map scenario index={index}')
        scenario_seed = generator_seed + index
        pair_group = f'unknown_{split}_pair_{index + 1:04d}'
        for profile in UNKNOWN_MAP_PROFILES:
            profile_record = records['profiles'][profile]
            obstacles = profile_record['obstacles']
            knowledge = _UNKNOWN_PROFILE_KNOWLEDGE[profile]
            family = _UNKNOWN_PROFILE_FAMILY[profile]
            code = PROFILE_CODES[profile]
            manifests[profile]['scenarios'].append({'scenario_id': f'unknown_{split}_{code}_{index + 1:04d}', 'pair_group_id': pair_group, 'scenario_profile': profile, 'scenario_role': split, 'scenario_split': split, 'protocol': CH3_UNKNOWN_MAP_V1, 'scenario_seed': int(scenario_seed), 'planner_seed': int(records['planner_seed']), 'max_steps': 400, 'target_motion_known': True, 'target_motion_mode': 'constant_velocity_reflect_v1', 'target_state_schema': 'moving_target_state_v1', 'target_motion_seed': int(generator_seed * 10 + index), 'initial_agent_positions': records['agents'].round(7).tolist(), 'initial_executor_wait_point': records['wait'].round(7).tolist(), 'target_position': records['target'].round(7).tolist(), 'target_initial_position': records['target'].round(7).tolist(), 'target_initial_velocity': records['velocity'].round(7).tolist(), 'use_obstacles': bool(obstacles), 'obstacle_layout_id': 'custom_aabb_v1' if obstacles else 'none', 'obstacle_family': family, 'obstacle_knowledge_mode': knowledge, 'obstacle_layout_sha256': _canonical_sha(obstacles), 'obstacle_count': len(obstacles), 'obstacles': deepcopy(obstacles), 'planner_grid_size': list(PLANNER_GRID_SIZE), 'planner_obstacle_clearance': PLANNER_OBSTACLE_CLEARANCE, 'target_obstacle_clearance': TARGET_OBSTACLE_CLEARANCE, 'target_capture_radius': TARGET_CAPTURE_RADIUS, 'target_trajectory_sha256': profile_record['trajectory_sha'], 'target_trajectory_distance': profile_record['trajectory_distance'], 'unknown_map_schema': 'shared_logodds_occupancy_v1', 'map_sharing_mode': 'central_shared_deterministic_v1', 'flow_phase_x': 0.0, 'flow_phase_y': 0.0, 'scenario_schema_version': 1, **profile_record['reachability']})
    _BUILD_CACHE[cache_key] = deepcopy(manifests)
    return deepcopy(manifests)
_DEFAULT_UNKNOWN_GENERATOR_SEEDS = {'train': UNKNOWN_TRAIN_GENERATOR_SEED, 'validation': UNKNOWN_VALIDATION_GENERATOR_SEED, 'smoke_train': UNKNOWN_SMOKE_TRAIN_GENERATOR_SEED, 'smoke_validation': UNKNOWN_SMOKE_VALIDATION_GENERATOR_SEED}

def build_scenario_manifests(count=DEFAULT_VALIDATION_SCENARIO_COUNT, generator_seed=None, split='train', profiles=None):
    selected = ['M20_MOVING_UNKNOWN_MULTI'] if profiles is None else list(profiles) if not isinstance(profiles, str) else [profiles]
    if selected != ['M20_MOVING_UNKNOWN_MULTI']:
        raise ValueError('active extraction generates only M20_MOVING_UNKNOWN_MULTI')
    if generator_seed is None:
        try:
            seed = int(_DEFAULT_UNKNOWN_GENERATOR_SEEDS[str(split)])
        except KeyError as exc:
            raise ValueError('unsupported active scenario split') from exc
    else:
        seed = int(generator_seed)
    manifests = _build_unknown_profile_manifests(count, seed, split)
    return {'M20_MOVING_UNKNOWN_MULTI': manifests['M20_MOVING_UNKNOWN_MULTI']}
