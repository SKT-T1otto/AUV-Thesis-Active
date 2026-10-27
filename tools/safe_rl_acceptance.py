"""Acceptance for curated recoverable mechanisms, not general performance."""
import copy,hashlib,json
import numpy as np
from tools.audit_safe_speed import declared_specs


def near_buffer_scene(parent):
    scene=copy.deepcopy(parent)
    positions=[[x,y,4.5] for x,y in ((3.999,3.),(3.999,11.),(11.999,3.),(11.999,11.))]
    obstacles=[dict(center=[p[0]+.5015,p[1],p[2]],size=[1.,1.2,2.]) for p in positions]
    scene.update(scenario_id='buffer_exit_four_roles_v2',initial_agent_positions=positions,
        initial_executor_wait_point=[16.,16.,4.5],target_position=[17.,17.,4.5],target_initial_position=[17.,17.,4.5],
        obstacles=obstacles,obstacle_count=4,use_obstacles=True,
        obstacle_layout_sha256=hashlib.sha256(json.dumps(obstacles,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest())
    # This is a new stress fixture, not the frozen generated scene/trajectory.
    for key in list(scene):
        if key.startswith('target_trajectory') or 'reachability' in key:scene.pop(key)
    return scene


def require_recoverable_case(summary,frames,initial_positions,*,near_buffer=False):
    if summary.get('termination_reason')=='obstacle_collision' or summary.get('collision_records') or summary.get('first_collision_agent_ids'):
        raise ValueError('recoverable safety case collided')
    steps=len(frames)
    if steps!=32:raise ValueError('recoverable mechanism must complete all32 steps')
    trace=summary['safety_rl']['trace']
    expected={(t,i) for t in range(steps) for i in range(4)}
    if len(trace)!=4*steps or {(r['step'],r['agent_id']) for r in trace}!=expected:
        raise ValueError('incomplete all-role physical-step trace')
    initial=np.asarray(initial_positions,float);positions=np.array([initial,*[f['positions'] for f in frames]],float)
    if positions.shape!=(steps+1,4,3) or not np.isfinite(positions).all():raise ValueError('invalid physical frames')
    for r in trace:
        if r['mode'].startswith('UNCERTIFIED') and r['model_checked']:raise ValueError('uncertified control mislabeled')
        if not np.isfinite(r['executed_acceleration']).all():raise ValueError('nonfinite control')
        s=declared_specs()[r['agent_id']]
        if np.any(np.abs(r['executed_acceleration'])>np.array([s['a_xy_max'],s['a_xy_max'],s['a_z_max']])+1e-6):raise ValueError('control exceeds original physical limits')
    travel=np.linalg.norm(np.diff(positions,axis=0),axis=2).sum(0)
    displacement=np.linalg.norm(positions[-1]-positions[0],axis=1)
    if np.any(travel<.15) or np.any(displacement<.1):raise ValueError('safety case stalled instead of recovering')
    if near_buffer:
        for i in range(4):
            if positions[1,i,0]>=initial[i,0]-1e-4:raise ValueError('near-buffer first motion did not exit toward free side')
            if initial[i,0]-positions[-1,i,0]<.52:raise ValueError('near-buffer recovery stopped inside the original margin')
            if not any(r['agent_id']==i and r['mode']=='BUFFER_EXIT' and r['model_checked'] for r in trace):raise ValueError('near-buffer role not exercised')
    return dict(passed=True,collision_free=True,all_roles_checked=True,path_length=travel.tolist(),net_displacement=displacement.tolist(),
        near_buffer_all_roles=near_buffer,scope='32-step curated recoverable mechanisms only; no full-episode safety or task-success claim')
