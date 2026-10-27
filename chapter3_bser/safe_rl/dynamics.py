"""Public-model predictions; no environment steps, truth geometry or flow phase."""
import math
import numpy as np


def limits(spec):
    return np.array([spec['a_xy_max'],spec['a_xy_max'],spec['a_z_max']],float)


def prior(position,velocity,target,spec,contract,*,speed_xy=None,speed_z=None):
    p,v,g=map(lambda x:np.asarray(x,float),(position,velocity,target))
    delta=g-p;distance=max(float(np.linalg.norm(delta[:2])),float(contract['eps']))
    speed_xy=spec['v_xy_max'] if speed_xy is None else speed_xy
    speed_z=spec['v_z_max'] if speed_z is None else speed_z
    desired=np.empty(3)
    desired[:2]=delta[:2]/distance*speed_xy*min(distance/contract['prior_slow_radius_xy'],1.)
    desired[2]=np.clip(delta[2]/contract['prior_slow_radius_z'],-1.,1.)*speed_z
    gain=np.array([contract['prior_kv_xy']]*2+[contract['prior_kv_z']])
    return np.clip(gain*(desired-v),-limits(spec),limits(spec))


def nominal(position,velocity,target,action,spec,contract,agent_id):
    kind='search' if agent_id<3 else 'executor'
    base=prior(position,velocity,target,spec,contract) if contract['use_residual_prior'] else np.zeros(3)
    base=base*contract['prior_strength_'+kind]
    residual=contract['residual_scale_'+kind]*limits(spec)*np.clip(action,-1.,1.)
    return np.clip(base+residual,-limits(spec),limits(spec)),base,residual


def integrate(position,velocity,acceleration,spec,dt,flow=(0.,0.)):
    """Same acceleration->drag/flow->speed clip->position order as original core."""
    v=np.asarray(velocity,float).copy();a=np.clip(acceleration,-limits(spec),limits(spec))
    v+=a*dt
    v[:2]+=(-spec['drag_xy']*v[:2]+np.asarray(flow,float))*dt
    v[2]+=(-spec['drag_z']*v[2]+spec['buoyancy_bias'])*dt
    speed=float(np.linalg.norm(v[:2]))
    if speed>spec['v_xy_max']:v[:2]*=spec['v_xy_max']/speed
    v[2]=np.clip(v[2],-spec['v_z_max'],spec['v_z_max'])
    return np.asarray(position,float)+v*dt,v


def brake(velocity,spec,dt,flow=(0.,0.)):
    # Deadbeat when feasible, otherwise use the ORIGINAL full physical bounds.
    v=np.asarray(velocity,float);rxy=1-spec['drag_xy']*dt;rz=1-spec['drag_z']*dt
    a=np.empty(3);a[:2]=-v[:2]/dt-np.asarray(flow,float)/rxy
    norm=float(np.linalg.norm(a[:2]))
    if norm>spec['a_xy_max']:a[:2]*=spec['a_xy_max']/norm
    a[2]=-v[2]/dt-spec['buoyancy_bias']/rz
    return np.clip(a,-limits(spec),limits(spec))


def stopping_bound(speed,spec,dt,flow_bound,*,vertical=False,reaction_steps=1):
    """Conservative travel budget, deliberately discarding helpful drag.

    Includes one reaction interval and an extra discretization interval. This
    is a MODEL bound for speed governance, not a geometric sensing certificate.
    Buoyancy is included as an adverse disturbance for either vertical direction.
    """
    speed=max(0.,float(speed))
    drag=spec['drag_z' if vertical else 'drag_xy']
    a=spec['a_z_max' if vertical else 'a_xy_max']*(1-drag*dt)
    disturbance=abs(spec['buoyancy_bias']) if vertical else math.sqrt(2)*flow_bound
    effective=a-disturbance
    if effective<=0:return float('inf')
    return speed*dt*(reaction_steps+1)+speed*speed/(2*effective)


def speed_limit(distance,spec,dt,flow_bound,*,vertical=False,reaction_steps=1):
    lo,hi=0.,float(spec['v_z_max' if vertical else 'v_xy_max'])
    if distance<=0:return 0.
    for _ in range(32):
        mid=(lo+hi)/2
        if stopping_bound(mid,spec,dt,flow_bound,vertical=vertical,reaction_steps=reaction_steps)<=distance:lo=mid
        else:hi=mid
    return lo


def speed_audit(specs,contract,options):
    dt=float(contract['dt']);distance=float(contract['obstacle_sensor_range'])
    rows=[]
    for i,spec in enumerate(specs):
        budget=distance-options['clearance']-options['hit_padding']
        horizontal=stopping_bound(spec['v_xy_max'],spec,dt,options['flow_component_bound'],reaction_steps=options['reaction_steps'])
        vertical=stopping_bound(spec['v_z_max'],spec,dt,options['flow_component_bound'],vertical=True,reaction_steps=options['reaction_steps'])
        rows.append(dict(agent_id=i,name=spec['name'],physical_v_xy_max=spec['v_xy_max'],
            full_sensor_range=distance,available_distance_after_margin=budget,
            horizontal_reaction_and_stop_bound=horizontal,vertical_reaction_and_stop_bound=vertical,
            independent_axis_bounds_fit_range=horizontal<=budget and vertical<=budget,
            horizontal_speed_cap_at_full_range=speed_limit(budget,spec,dt,options['flow_component_bound'],reaction_steps=options['reaction_steps']),
            distance_at_20_step_refresh=spec['v_xy_max']*20*dt,
            scope='Conservative independent-axis model audit; not proof of ray coverage or combined 3D trajectory safety'))
    return rows
