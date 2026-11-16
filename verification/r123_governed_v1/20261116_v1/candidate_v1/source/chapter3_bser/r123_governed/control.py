import numpy as np
from chapter3_bser.r123.options import CONTROL
from chapter3_bser.reach_avoid.goals import Goal,base_action
from chapter3_bser.reach_avoid.options import SPECS,CONTRACT
from chapter3_bser.reach_avoid.scan_geometry import refined_buffer_exit
from chapter3_bser.safe_rl.dynamics import prior,brake,integrate,limits


def governed_base(packet,i,track,caps):
    """Limit desired/one-step speed, retaining actual acceleration constraints."""
    s=SPECS[i];p=packet.positions[i];v=packet.velocities[i];dt=CONTRACT['dt']
    flow=packet.flow[i]
    u=prior(p,v,track,s,CONTRACT,speed_xy=caps[0],speed_z=caps[1])
    if packet.phases[i]==2:
        u+=np.array([CONTRACT['prior_kv_xy']]*2+[CONTRACT['prior_kv_z']])*packet.target_velocity
    u[:2]-=flow/(1-s['drag_xy']*dt)
    u[2]-=s['buoyancy_bias']/(1-s['drag_z']*dt)
    # Public dynamics order: acceleration, drag+flow, speed clip. Project only
    # the desired next velocity, then invert that order and clip real authority.
    nv=integrate(p,v,u,s,dt,flow)[1]
    speed=np.linalg.norm(nv[:2])
    if speed>caps[0]:nv[:2]*=caps[0]/max(speed,1e-30)
    nv[2]=np.clip(nv[2],-caps[1],caps[1])
    out=np.empty(3)
    out[:2]=((nv[:2]-flow*dt)/(1-s['drag_xy']*dt)-v[:2])/dt
    out[2]=((nv[2]-s['buoyancy_bias']*dt)/(1-s['drag_z']*dt)-v[2])/dt
    return np.clip(out/limits(s),-1,1).astype(np.float32)


def conditional_banks(packet,i,tracks,geometry,guard,r1=True):
    if not r1:raise ValueError('governed rule uses the existing R1 residual domain')
    s=SPECS[i];p=packet.positions[i];v=packet.velocities[i];flow=packet.flow[i];scale=limits(s)
    caps=guard._caps(p,v,s,geometry);violation=max(0.,np.linalg.norm(v[:2])-caps[0])+max(0.,abs(v[2])-caps[1])
    b=brake(v,s,CONTRACT['dt'],flow)/scale
    exit_result=refined_buffer_exit(p,v,s,geometry,guard.contract,guard.options,flow)
    banks=[];masks=[];forces=[];emergencies=[];diagnostics=[];checked={}
    for track in tracks:
        base=governed_base(packet,i,track,caps)
        a=np.zeros((16,3),np.float32);a[0]=base
        for k,(axis,sign) in enumerate(((axis,sign) for axis in range(3) for sign in (-1,1)),1):
            a[k]=np.clip(base+np.eye(3)[axis]*sign*CONTROL['residual_bound'],-1,1)
        a[7]=base;a[8]=b;a[9]=b if exit_result is None else exit_result[0]/scale
        valid=np.zeros(16,bool);details=[]
        for k in (*range(7),8):
            key=a[k].tobytes()
            if key not in checked:
                nv=integrate(p,v,a[k]*scale,s,CONTRACT['dt'],flow)[1]
                nv_bad=max(0.,np.linalg.norm(nv[:2])-caps[0])+max(0.,abs(nv[2])-caps[1])
                speed_ok=nv_bad<=1e-6 or nv_bad<violation-1e-8
                ok,probe=guard.check(p,v,a[k]*scale,s,geometry,flow) if speed_ok else (False,dict(reason='speed_cap'))
                checked[key]=(bool(ok),dict(speed_ok=bool(speed_ok),next_speed_xy=float(np.linalg.norm(nv[:2])),next_speed_z=float(abs(nv[2])),**probe))
            valid[k],detail=checked[key];details.append(dict(action=k,**detail))
        valid[9]=exit_result is not None
        needs_motion=np.linalg.norm(np.asarray(track)-p)>.05
        moving=any(valid[k] and np.linalg.norm(integrate(p,v,a[k]*scale,s,CONTRACT['dt'],flow)[1])>1e-6 for k in range(7))
        navigation=bool(valid[:7].any()) and (not needs_motion or moving)
        if not navigation:valid[:7]=False
        if navigation:valid[8]=False
        emergency=not bool(valid.any())
        if emergency:valid[8]=True
        forced=9 if exit_result is not None else 8 if not navigation else -1
        banks.append(a);masks.append(valid);forces.append(forced);emergencies.append(emergency)
        nominal=base_action(packet,i,Goal(tuple(track),tuple(packet.target_velocity) if packet.phases[i]==2 else (0.,0.,0.)))
        diagnostics.append(dict(speed_cap_xy=float(caps[0]),speed_cap_z=float(caps[1]),available_distance=float(caps[2]),
            nominal_normalized=nominal.tolist(),governed_normalized=base.tolist(),action_mask=valid.tolist(),
            navigation_candidates=int(valid[:7].sum()),navigation_blocked=not navigation,verified_exit=exit_result is not None,
            emergency=emergency,candidate_checks=details))
    return np.array(banks),np.array(masks),np.array(forces),np.array(emergencies),diagnostics
