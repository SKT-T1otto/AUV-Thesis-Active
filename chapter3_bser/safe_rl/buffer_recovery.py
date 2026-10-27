"""Local egress from an already violated public margin; no truth queries.

Ordinary segment checks remain strict. This separate finite check permits
motion away from the initial offending surfaces, never toward them. Constant
bounded-flow corner probes are a stress check, not an invariant certificate.
"""
from itertools import product
import numpy as np
from chapter3_bser.safe_rl.dynamics import integrate,brake,limits


def _touches(p,q,lo,hi):
    if not len(lo):return False
    d=q-p;near=np.full_like(lo,-np.inf);far=np.full_like(lo,np.inf);outside=np.zeros(len(lo),bool)
    for j in range(3):
        if abs(d[j])<1e-15:outside|=(p[j]<lo[:,j])|(p[j]>hi[:,j])
        else:
            a,b=(lo[:,j]-p[j])/d[j],(hi[:,j]-p[j])/d[j]
            near[:,j]=np.minimum(a,b);far[:,j]=np.maximum(a,b)
    return bool(np.any(~outside & (np.maximum(0.,near.max(1))<=np.minimum(1.,far.min(1)))))


class ExitCorridor:
    def __init__(self,geometry,origin,extra_margin=0.):
        self.g=geometry;self.origin=np.asarray(origin,float);p=self.origin
        margin=geometry.padding+extra_margin
        lo,hi=geometry.raw.occupied_lower,geometry.raw.occupied_upper
        closest=np.clip(p,lo,hi);delta=p-closest;dist=np.linalg.norm(delta,axis=1)
        self.enclosing=dist<1e-10
        # Ordinary voxel margins are axis-aligned boxes, not Euclidean balls.
        # Classify corner/edge overlaps using the SAME padded-box geometry.
        near=np.all((p>=lo-margin)&(p<=hi+margin),axis=1)&~self.enclosing
        self.normals=list(delta[near]/dist[near,None])
        hd=p-geometry.points;distance=np.linalg.norm(hd,axis=1);hit_near=distance<=margin
        self.valid=not np.any(distance<1e-8)
        # Coarse cell occupation can include an otherwise valid physical pose.
        # It is ambiguous; require directional scan evidence to attempt egress.
        if np.any(self.enclosing) and not np.any(hit_near):self.valid=False
        self.normals.extend(hd[hit_near]/np.maximum(distance[hit_near,None],1e-10))
        self.normals=np.array(self.normals,float).reshape(-1,3)
        self.far_lo=lo[~(near|self.enclosing)]-margin
        self.far_hi=hi[~(near|self.enclosing)]+margin
        self.raw_lo,self.raw_hi=lo[~self.enclosing],hi[~self.enclosing]
        self.far_hits=geometry.points[~hit_near]

    def segment(self,p,q):
        g=self.g;p,q=np.asarray(p,float),np.asarray(q,float)
        if not self.valid or not len(self.normals):return False
        if not np.isfinite([p,q]).all() or np.any(np.minimum(p,q)<g.raw.lower) or np.any(np.maximum(p,q)>g.raw.upper):return False
        # Supporting planes through the INITIAL pose forbid approaching any
        # initially near evidence. Raw cells remain closed forbidden volumes.
        if np.any(self.normals@(p-self.origin)<-1e-9) or np.any(self.normals@(q-self.origin)<-1e-9):return False
        if _touches(p,q,self.raw_lo,self.raw_hi) or _touches(p,q,self.far_lo,self.far_hi):return False
        if len(self.far_hits):
            d=q-p;t=np.clip((self.far_hits-p)@d/max(float(d@d),1e-30),0.,1.)
            if np.any(np.linalg.norm(self.far_hits-(p+t[:,None]*d),axis=1)<=g.padding):return False
        return True


def velocity_command(v,desired,spec,dt,flow):
    v=np.asarray(v,float);desired=np.asarray(desired,float)
    u=(desired-v)/dt
    u[:2]-=np.asarray(flow)/(1-spec['drag_xy']*dt)
    u[2]-=spec['buoyancy_bias']/(1-spec['drag_z']*dt)
    # Use full original bounds; the requested exit speed is small. No global
    # physical limit, reward or task residual contract is changed.
    return np.clip(u,-limits(spec),limits(spec))


def buffer_exit(p,v,spec,geometry,contract,options,flow):
    # Overlap the ordinary backup envelope so that a partially completed exit
    # does not stall immediately outside the nominal .52m margin.
    if geometry.distance(p)>geometry.padding+.10:return None
    corridor=ExitCorridor(geometry,p,extra_margin=.10)
    if not corridor.valid or not len(corridor.normals):return None
    dt=contract['dt'];bound=options['flow_component_bound'];origin=np.asarray(p,float)
    directions=[corridor.normals.sum(0),*corridor.normals,*np.eye(3),*-np.eye(3)]
    flows=[np.asarray(flow,float),*[np.array(x) for x in product((-bound,bound),repeat=2)]]
    for direction in directions:
        norm=np.linalg.norm(direction)
        if norm<1e-8:continue
        direction=direction/norm
        if np.any(corridor.normals@direction<1e-8):continue
        desired=.35*direction;command=velocity_command(v,desired,spec,dt,flow)
        passed=True;details=[]
        for disturbance in flows:
            x=np.asarray(p,float).copy();vel=np.asarray(v,float).copy();exited=False;stopped=False
            for k in range(options['horizon']):
                if geometry.distance(x)>geometry.padding+.15:exited=True
                u=command if k==0 else brake(vel,spec,dt,disturbance) if exited else velocity_command(vel,desired,spec,dt,disturbance)
                nxt,nv=integrate(x,vel,u,spec,dt,disturbance)
                if not corridor.segment(x,nxt):passed=False;break
                if k==0:
                    # Exempt zero isotropic cap only for strictly outward motion
                    # under every first-step disturbance corner.
                    if np.min(corridor.normals@(nxt-origin))<=1e-7 or np.linalg.norm(nv[:2])>.5 or abs(nv[2])>.4:
                        passed=False;break
                x,vel=nxt,nv
                if exited and np.linalg.norm(vel)<.04:stopped=True;break
            if not passed or not stopped:passed=False;break
            details.append(dict(flow=disturbance.tolist(),steps=k+1,stop=x.tolist()))
        if passed:return command,dict(reason='outward_margin_exit_and_stop',flow_probe_model='constant bounded XY flow, feedback after first step',
            probes=details,origin_in_occupied_voxel=bool(np.any(corridor.enclosing)),terminal_invariant_certified=False,
            ground_truth_geometry_used=False,isotropic_zero_cap_exemption=True)
    return None
