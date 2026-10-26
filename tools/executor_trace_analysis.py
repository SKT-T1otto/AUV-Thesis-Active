"""Independent posthoc geometry and motion checks. No runtime construction."""
import csv
import math
from pathlib import Path
import numpy as np
from tools.executor_trace import Blobs, read, write, plain, verified_recording


def slab(start, end, lower, upper):
    values = [np.asarray(x, dtype=float) for x in (start, end, lower, upper)]
    if any(v.shape != (3,) or not np.isfinite(v).all() for v in values):
        raise ValueError('finite 3D geometry required')
    start, end, lower, upper = values
    if np.any(lower > upper):
        raise ValueError('invalid AABB')
    enter, leave = 0., 1.
    for a, d, lo, hi in zip(start, end-start, lower, upper):
        if d == 0:
            if a < lo or a > hi:
                return None
        else:
            left, right = sorted(((lo-a)/d, (hi-a)/d))
            enter, leave = max(enter, left), min(leave, right)
            if enter > leave:
                return None
    return enter


def point_clearance(point, lower, upper):
    p, lo, hi = (np.asarray(x, dtype=float) for x in (point, lower, upper))
    return float(np.linalg.norm(np.maximum(np.maximum(lo-p, p-hi), 0)))


def segment_clearance(start, end, lower, upper):
    a, b, lo, hi = (np.asarray(x, dtype=float) for x in (start, end, lower, upper))
    delta, times = b-a, {0., 1.}
    for p, d, l, h in zip(a, delta, lo, hi):
        if d:
            times.update(t for t in ((l-p)/d, (h-p)/d) if 0 < t < 1)
    times = sorted(times)
    candidates = list(times)
    for left, right in zip(times, times[1:]):
        middle = a+(left+right)*.5*delta
        active = (middle < lo) | (middle > hi)
        bound = np.where(middle < lo, lo, hi)
        denominator = float(np.sum(delta[active]**2))
        if denominator:
            t = float(-np.sum(delta[active]*(a-bound)[active])/denominator)
            candidates.append(min(right, max(left, t)))
    return min(point_clearance(a+t*delta, lo, hi) for t in candidates)


def path_geometry(points, lower, upper):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not len(points):
        return dict(available=False, min_clearance=None, intersections=[])
    if len(points) == 1:
        points = np.repeat(points, 2, axis=0)
    hits, clearance = [], math.inf
    for index, (a, b) in enumerate(zip(points, points[1:])):
        t = slab(a, b, lower, upper)
        if t is not None:
            hits.append(dict(segment=index, fraction=t))
        clearance = min(clearance, segment_clearance(a, b, lower, upper))
    return dict(available=True, min_clearance=clearance, intersections=hits)


def polyline_distance(point, points):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    p = np.asarray(point)
    if not len(points):
        return None
    if len(points) == 1:
        return float(np.linalg.norm(p-points[0]))
    distances = []
    for a, b in zip(points, points[1:]):
        d = b-a
        t = 0. if not np.dot(d, d) else np.clip(np.dot(p-a, d)/np.dot(d, d), 0., 1.)
        distances.append(float(np.linalg.norm(p-a-t*d)))
    return min(distances)


def boxes(scene):
    return [(np.asarray(o['center'])-np.asarray(o['size'])/2,
             np.asarray(o['center'])+np.asarray(o['size'])/2) for o in scene['obstacles']]


def public_segment(start, end, grid, occupied, unknown, *, clearance=0.):
    centers, spacing = np.asarray(grid['cell_centers']), np.asarray(grid['spacing'])
    occupied, unknown = np.asarray(occupied).reshape(-1), np.asarray(unknown).reshape(-1)
    touched = [i for i, c in enumerate(centers) if slab(start, end, c-spacing/2, c+spacing/2) is not None]
    padded = [int(i) for i in np.flatnonzero(occupied)
              if slab(start, end, centers[i]-spacing/2-clearance, centers[i]+spacing/2+clearance) is not None]
    return dict(touched_cells=touched, occupied_cells=[i for i in touched if occupied[i]],
        unknown_cells=[i for i in touched if unknown[i]], padded_occupied_cells=padded,
        classification='independent exact closed voxel oracle, not the original sampled predicate')


def sampled_segment(start, end, live):
    """Reproduce the stored OLD sample formula without calling planner/cache APIs."""
    a, b = np.asarray(start), np.asarray(end)
    spacing = np.asarray([live['cell_dx'], live['cell_dy'], live['cell_dz']])
    shape, eps = np.asarray(live['grid_size']), float(live['eps'])
    sample_step = max(.15, .4*min(spacing[0], spacing[1], spacing[2] if spacing[2] > eps else min(spacing[:2])))
    count = max(1, int(math.ceil(float(np.linalg.norm(b-a))/sample_step)))
    points = a+np.linspace(0, 1, count+1)[:, None]*(b-a)
    cells = np.empty(points.shape, dtype=np.int64)
    cells[:, :2] = np.floor(points[:, :2]/np.maximum(spacing[:2], eps))
    cells[:, 2] = np.rint((points[:, 2]-live['z_range'][0])/max(spacing[2], eps))
    cells = np.clip(cells, 0, shape-1)
    flats = sorted(set(int(x) for x in np.ravel_multi_index(cells.T, shape)))
    occupied = np.asarray(live['known_occupied_mask']).reshape(-1)
    bounds = np.asarray(live['_space_size_np'])
    bounded = bool(np.all(a >= 0) and np.all(b >= 0) and np.all(a <= bounds) and np.all(b <= bounds))
    return dict(sampled_cells=flats, occupied_samples=[i for i in flats if occupied[i]],
                accepted=bounded and not bool(occupied[flats].any()))


def nominal_acceleration(position, velocity, target, spec, control):
    p, v, target = (np.asarray(x, dtype=float) for x in (position, velocity, target))
    delta = target-p
    norm = max(float(np.linalg.norm(delta[:2])), control['eps'])
    desired_xy = delta[:2]/norm*spec['v_xy_max']*min(norm/control['prior_slow_radius_xy'], 1.)
    acc = np.empty(3)
    acc[:2] = np.clip(control['prior_kv_xy']*(desired_xy-v[:2]), -spec['a_xy_max'], spec['a_xy_max'])
    desired_z = np.clip(delta[2]/control['prior_slow_radius_z'], -1, 1)*spec['v_z_max']
    acc[2] = np.clip(control['prior_kv_z']*(desired_z-v[2]), -spec['a_z_max'], spec['a_z_max'])
    return acc*control['prior_strength_executor'] if control['use_residual_prior'] else np.zeros(3)


def velocity_step(velocity, acceleration, spec, dt, disturbance=(0., 0.)):
    v = np.asarray(velocity, dtype=float)+np.asarray(acceleration)*dt
    v[:2] += (-spec['drag_xy']*v[:2]+np.asarray(disturbance))*dt
    v[2] += (-spec['drag_z']*v[2]+spec['buoyancy_bias'])*dt
    speed = float(np.linalg.norm(v[:2]))
    if speed > spec['v_xy_max']:
        v[:2] *= spec['v_xy_max']/speed
    v[2] = np.clip(v[2], -spec['v_z_max'], spec['v_z_max'])
    return v


def motion_evidence(frame, contract):
    physics = frame['physical_step']
    before, after = physics['before']['physical'], physics['after']
    p, v, target = (np.asarray(before[k][3]) for k in ('_agent_pos', '_agent_vel', '_nav_targets'))
    spec, control = contract['specs'][3], contract['control']
    dt = control['dt']
    hit = next((c for c in physics['collisions'] if c['agent_id'] == 3), None)
    end = np.asarray(hit['candidate_position'] if hit else after['_agent_pos'][3])
    candidate_velocity = (end-p)/dt
    acc = np.asarray(after['_agent_acc'][3])
    expected_prior = nominal_acceleration(p, v, target, spec, control)
    no_flow_velocity = velocity_step(v, acc, spec, dt)
    speed_clipped = bool(np.linalg.norm(candidate_velocity[:2]) >= spec['v_xy_max']-1e-5
                         or abs(candidate_velocity[2]) >= spec['v_z_max']-1e-5)
    at_wall = bool(np.any(end <= 0) or np.any(end >= np.asarray(contract['bounds'])))
    infer_ok = not speed_clipped and not at_wall
    # Horizontal Euler flow is identifiable only before speed/wall clipping.
    inferred_flow = ((candidate_velocity[:2]-(v[:2]+acc[:2]*dt)*(1-spec['drag_xy']*dt))/dt).tolist() if infer_ok else None
    return dict(actual_tracking_target=target.tolist(), velocity_before=v.tolist(),
        actual_prior_acceleration=after['_last_prior_acc'][3],
        prior_reconstruction_max_abs_error=float(np.max(np.abs(expected_prior-np.asarray(after['_last_prior_acc'][3])))),
        actual_residual_acceleration=after['_last_residual_acc'][3],
        actual_acceleration=acc.tolist(), candidate_velocity=candidate_velocity.tolist(),
        no_flow_one_step_velocity=no_flow_velocity.tolist(),
        inferred_horizontal_disturbance=inferred_flow, disturbance_identifiable=infer_ok,
        disturbance_scope='inferred from recorded transition; includes float/integration error; not hidden flow truth',
        speed_clipped=speed_clipped, at_wall=at_wall,
        actual_navigation_matches_installed_tracking=bool(np.allclose(target, frame['before']['executor_guidance']['tracking_waypoint'], atol=1e-6, rtol=0)),
        learned_residual_zero=bool(np.all(np.asarray(after['_last_residual_acc'][3]) == 0)))


def braking_counterfactual(frame, contract, obstacles, *, steps=16):
    evidence = motion_evidence(frame, contract)
    before = frame['physical_step']['before']['physical']
    p, v = np.asarray(before['_agent_pos'][3], float), np.asarray(before['_agent_vel'][3], float)
    anchor, spec, control = p.copy(), contract['specs'][3], contract['control']
    flow = evidence['inferred_horizontal_disturbance']
    if flow is None:
        return dict(available=False, reason='clipped transition prevents disturbance identification')
    minimum, first = math.inf, None
    for i in range(steps):
        acc = nominal_acceleration(p, v, anchor, spec, control)
        v = velocity_step(v, acc, spec, control['dt'], flow)
        end = p+v*control['dt']
        for j, (lo, hi) in enumerate(obstacles):
            minimum = min(minimum, segment_clearance(p, end, lo, hi))
            if first is None and slab(p, end, lo, hi) is not None:
                first = dict(predicted_step=i+1, obstacle_id=j)
        p = end
    return dict(available=True, first_collision=first, min_true_clearance=None if not obstacles else minimum,
        final_speed=float(np.linalg.norm(v)), steps=steps,
        scope='OFFLINE fixed-anchor nominal brake with frozen inferred disturbance; ground truth only after close; not a rerun or proof of causality')


def frame_evidence(frame, blobs, obstacle, contract):
    before = frame['before']
    physical = frame['physical_step']['before']['physical']
    p = np.asarray(physical['_agent_pos'][3])
    target = np.asarray(before['semantic_assignment']['target_region'])
    tracking = np.asarray(physical['_nav_targets'][3])
    lo, hi = obstacle
    semantic_path = before['semantic_assignment']['path']
    remaining_original = before['tracker']['remaining_path_points']
    state, live = blobs.get(before['decision_state_blob']), blobs.get(before['live_map_blob'])
    margin = float(live.get('planner_obstacle_clearance', .4))
    maps = {}
    for label, grid, occupied, unknown in (
        ('live',dict(cell_centers=live['flat_xyz_centers'],spacing=[live['cell_dx'],live['cell_dy'],live['cell_dz']]),live['known_occupied_mask'],live['unknown_mask']),
        ('cached',state['grid'],state['occupancy']['occupied_mask'],state['occupancy']['unknown_mask'])):
        maps[label] = public_segment(p, tracking, grid, occupied, unknown, clearance=margin)
    old_sampled = sampled_segment(p, tracking, live)
    actual_hit = next((h for h in frame['physical_step']['collisions'] if h['agent_id'] == 3), None)
    end = actual_hit['candidate_position'] if actual_hit else frame['physical_step']['after']['_agent_pos'][3]
    controller = frame.get('controller_before_step')
    detection = {} if controller is None else controller.get('event_detection', {})
    return dict(step_before=frame['step_before'], position=p.tolist(), semantic_target=target.tolist(),
        tracking_target=tracking.tolist(), distance_to_semantic_target=float(np.linalg.norm(p-target)),
        semantic_target_clearance=point_clearance(target,lo,hi), position_clearance=point_clearance(p,lo,hi),
        tracking_segment=path_geometry([p,tracking],lo,hi), actual_motion=path_geometry([p,end],lo,hi),
        full_semantic_path=path_geometry(semantic_path,lo,hi),
        remaining_original_path=path_geometry(remaining_original,lo,hi),
        remaining_with_current_connector=path_geometry([p.tolist(),*remaining_original],lo,hi),
        distance_to_original_polyline=polyline_distance(p,semantic_path),
        hold=before['executor_guidance']['hold_state'], reachable=before['executor_guidance']['reachable'],
        path_index=before['tracker']['next_index'], threshold=before['path_tracking_threshold'],
        cached_map_revision=state['map_revision'], live_map_revision=live['map_revision'],
        provider_refresh=before['provider_refresh'], controller_detection=detection,
        public_tracking=maps, sampled_live_tracking=old_sampled,
        guard_last=before['guard_last'], motion=motion_evidence(frame,contract))


def analyze_episode(directory, scene, output):
    """Called only after run_episode closes its environment and seals recording."""
    directory, output = Path(directory), Path(output)
    receipt = verified_recording(directory)
    if not receipt['instrumented']:
        raise ValueError('full trace required for causal evidence')
    output.mkdir(parents=True, exist_ok=False)
    frames = [__import__('json').loads(line) for line in (directory/'trace.jsonl').read_text(encoding='utf-8').splitlines()]
    contract = read(directory/'public_control.json')
    blobs, obstacles = Blobs(directory/'blobs',existing=True), boxes(scene)
    collision_frame = next((f for f in frames if any(h['agent_id']==3 for h in f['physical_step']['collisions'])),None)
    hit = None if collision_frame is None else next(h for h in collision_frame['physical_step']['collisions'] if h['agent_id']==3)
    # A non-collision run still has complete logs; never invent a collision cause.
    timeline = [] if hit is None else [frame_evidence(f,blobs,obstacles[hit['obstacle_id']],contract) for f in frames]
    result = dict(completed=True, executor_collision=hit is not None, collision=hit,
        causal_ranking_established=False, evidence_scope='observational facts plus explicitly labelled offline counterfactuals',
        ground_truth_used_only_after_runtime_close=True, physical_steps=receipt['physical_steps'])
    audits=[motion_evidence(f,contract) for f in frames]
    result['motion_audit']=dict(steps=len(audits),
        max_prior_reconstruction_error=max((a['prior_reconstruction_max_abs_error'] for a in audits),default=None),
        actual_navigation_mismatch_steps=[f['step_before'] for f,a in zip(frames,audits) if not a['actual_navigation_matches_installed_tracking']],
        nonzero_learned_residual_steps=[f['step_before'] for f,a in zip(frames,audits) if not a['learned_residual_zero']])
    if hit is not None:
        last, lohi = timeline[-1], obstacles[hit['obstacle_id']]
        first = lambda predicate: next((r['step_before'] for r in timeline if predicate(r)),None)
        result.update(collision_step=collision_frame['step_before']+1,
            collision_phase=collision_frame['physical_step']['before']['phase'], obstacle_lower=lohi[0].tolist(),obstacle_upper=lohi[1].tolist(),
            collision_frame=last,
            earliest_remaining_path_true_intersection=first(lambda r: bool(r['remaining_original_path']['intersections'])),
            earliest_live_tracking_occupied=first(lambda r: bool(r['public_tracking']['live']['occupied_cells'])),
            earliest_cached_tracking_occupied=first(lambda r: bool(r['public_tracking']['cached']['occupied_cells'])),
            earliest_tracking_sample_disagreement=first(lambda r: r['sampled_live_tracking']['accepted'] and bool(r['public_tracking']['live']['occupied_cells'])),
            earliest_hold=first(lambda r:r['hold']),
            last_16_steps=timeline[-16:],
            braking_counterfactuals=[dict(step_before=f['step_before'],**braking_counterfactual(f,contract,obstacles)) for f in frames[-16:]],
            hypotheses=dict(target_margin=dict(distance=last['distance_to_semantic_target'],clearance=last['semantic_target_clearance'],status='requires target distance and residence history; no threshold-based blame'),
                path_geometry=dict(remaining_path_intersects=bool(last['remaining_original_path']['intersections']),tracking_connector_intersects=bool(last['tracking_segment']['intersections'])),
                map_or_staleness=dict(live_occupied=last['public_tracking']['live']['occupied_cells'],cached_occupied=last['public_tracking']['cached']['occupied_cells'],unknown=last['public_tracking']['live']['unknown_cells'],detection=last['controller_detection']),
                tracking_motion=dict(distance_to_original_polyline=last['distance_to_original_polyline'],motion_intersects=bool(last['actual_motion']['intersections']),motion=last['motion']),
                braking_or_drift=dict(hold=last['hold'],velocity=last['motion']['velocity_before'],guard=last['guard_last'])))
    write(output/'analysis.json',result)
    with (output/'executor_timeline.csv').open('x',newline='',encoding='utf-8') as stream:
        names=('step_before','distance_to_semantic_target','semantic_target_clearance','position_clearance',
               'distance_to_original_polyline','hold','reachable','path_index','cached_map_revision','live_map_revision')
        writer=csv.DictWriter(stream,fieldnames=names);writer.writeheader()
        writer.writerows({k:r[k] for k in names} for r in timeline)
    return result
