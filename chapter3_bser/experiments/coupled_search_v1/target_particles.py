"""Batch free flights; possible contacts use the unchanged scalar integrator."""
import numpy as np
from core.env.target_motion import TargetState,advance_target_state


def propagate(positions,velocities,geometry,step,contract,budget):
    p=np.asarray(positions,float);v=np.asarray(velocities,float)
    c=contract;dt=c['dt'];bounds=np.asarray(c['bounds'])
    if c['target_motion_mode']=='static' or dt==0:return p.copy(),v.copy()
    if c['target_motion_mode']!='constant_velocity_reflect_v1':raise ValueError('unsupported target motion')
    budget.check()
    end=p+v*dt
    contacts=np.any((end<0)|(end>bounds)|(p<0)|(p>bounds),axis=1)
    segment_lo=np.minimum(p,end);segment_hi=np.maximum(p,end)
    obstacles=geometry.obstacles()
    for box in obstacles:
        center=np.asarray(box['center']);half=np.asarray(box['size'])/2+c['target_obstacle_clearance']
        # Broad phase is conservative, including tangencies and inside starts.
        contacts|=np.all((segment_hi>=center-half)&(segment_lo<=center+half),axis=1)
    out_v=v.copy()
    for i in np.flatnonzero(contacts):
        budget.check()
        result=advance_target_state(TargetState(p[i],v[i],step,c['target_motion_mode']),dt,bounds,obstacles,
            clearance=c['target_obstacle_clearance'],max_reflections=c['target_max_reflections_per_step'])
        end[i]=result.position;out_v[i]=result.velocity
    return np.clip(end,0,bounds),out_v
