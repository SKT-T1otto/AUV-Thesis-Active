"""Shared supervised velocity correction for a frozen command and nominal controller."""
from types import SimpleNamespace

import numpy as np
from .experiments.coupled_search_v1.execution_forecast import nominal_step
from .experiments.coupled_search_v1.numerics import Budget,swept_distances
from .experiments.coupled_search_v1.target_particles import propagate
from .experiments.coupled_search_v1.geometry_hypotheses import GeometryHypothesis


def frame(delta):
    norm=np.linalg.norm(delta)
    tangent=delta/norm if norm>1e-8 else np.array([1.,0.,0.])
    axis=np.array([0.,0.,1.]) if abs(tangent[2])<.95 else np.array([0.,1.,0.])
    side=np.cross(axis,tangent)
    side/=np.linalg.norm(side)
    return np.vstack([tangent,side,np.cross(tangent,side)])


def features(p,v,path,cursor,endpoint,spec,contract,occupied,spacing):
    p=np.asarray(p)
    v=np.asarray(v)
    path=np.asarray(path).reshape(-1,3)
    target=path[cursor] if cursor<len(path) else endpoint
    delta=target-p
    basis=frame(delta)
    projected=basis@v
    speed=np.linalg.norm(v)
    cosine=projected[0]/max(speed,1e-8)
    next_target=path[cursor+1] if cursor+1<len(path) else endpoint
    outgoing=next_target-target
    bend=1-(outgoing@basis[0])/max(1e-8,np.linalg.norm(outgoing)) if np.linalg.norm(outgoing)>1e-8 else 0.
    cell=np.clip(np.floor(p/spacing).astype(int),0,np.array(occupied.shape)-1)
    neighborhood=tuple(slice(max(0,k-1),k+2) for k in cell)
    local=float(np.mean(occupied[neighborhood]))
    cursor_fraction=cursor/max(1,len(path))
    physics=[spec[k] for k in ('v_xy_max','v_z_max','a_xy_max','a_z_max','drag_xy','drag_z','sensor_range')]
    vector=np.r_[physics,p/contract['bounds'],v,projected,speed,cosine,delta,bend,cursor_fraction,local]
    group=int(speed/spec['v_xy_max']>.25)+2*int(cosine<.7071)+4*int(local>0)+8*int(bend>.2929)
    return vector,group,basis,target


def regress(model,x,g):
    if model['kind']=='bins':
        return np.asarray(model['means'][g])
    return (x-np.asarray(model['center']))/np.asarray(model['scale'])@np.asarray(model['coeff'])+model['offset']


def fit(records,kind):
    x=np.array([v['x'] for v in records])
    y=np.array([v['y'] for v in records])
    group=np.array([v['g'] for v in records])
    base=y.mean(0)
    model=dict(kind=kind,means=[((y[group==g].sum(0)+8*base)/(sum(group==g)+8)).tolist() for g in range(16)])
    if kind=='ridge':
        scale=x.std(0)
        scale[scale<1e-6]=1
        norm=(x-x.mean(0))/scale
        coeff=np.linalg.solve(norm.T@norm+5*np.eye(x.shape[1]),norm.T@(y-base))
        model.update(center=x.mean(0).tolist(),scale=scale.tolist(),coeff=coeff.tolist(),offset=base.tolist())
    residual=y-np.array([regress(model,r['x'],r['g']) for r in records])
    global_var=np.mean(residual**2,axis=0)
    sigmas=np.array([np.sqrt((np.sum(residual[group==g]**2,axis=0)+8*global_var)/(sum(group==g)+8)+1e-8)
                     for g in range(16)])
    model['sigma']=sigmas.tolist()
    return model


def make_blocks(model,records):
    """Use whole synchronous physical steps only, never cross a family boundary."""
    by_family={}
    for r in records:
        mat=by_family.setdefault(r['family'],{}).setdefault(r['t'],{})
        mu=regress(model,r['x'],r['g'])
        mat[r['i']]=np.clip((r['y']-mu)/model['sigma'][r['g']],-3.,3.)
    blocks=[]
    provenance=[]
    for family,steps in sorted(by_family.items()):
        times=sorted(t for t,v in steps.items() if len(v)==3)
        runs=[]
        for t in times:
            if not runs or runs[-1][-1]+1!=t:
                runs.append([])
            runs[-1].append(t)
        selected=max(runs,key=len)
        start=max(0,(len(selected)-16)//2)
        take=selected[start:start+16]
        block=np.array([[steps[t][i] for i in range(3)] for t in take])
        blocks.append(block[np.arange(16)%len(block)].tolist())
        provenance.append(dict(family=family,physical_origins=take))
    model.update(blocks=blocks,block_sources=provenance)


def calibrated_route(model,item,spec,contract,occupied,spacing,geometry,blocked):
    path=np.asarray(item['path']).reshape(-1,3)
    endpoint=np.asarray(item['endpoint'])
    p0=np.asarray(item['pos'])
    i=item['id']
    blocks=np.asarray(model['blocks'])
    q,h=blocks.shape[:2]
    pos=np.tile(p0,(q,h+1,1))
    valid=np.zeros(q,dtype=int)
    if not blocked:
        for j in range(q):
            p,v=p0,np.asarray(item['vel'])
            cursor=item['cursor']
            for k in range(h):
                x,g,b,target=features(p,v,path,cursor,endpoint,spec,contract,occupied,spacing)
                local=regress(model,x,g)+np.asarray(model['sigma'][g])*blocks[j,k,i]
                error=b.T@local
                end,velocity=nominal_step(p,v,target,spec,contract,error=error)
                if not geometry.segment_free(p,end):
                    pos[j,k+1:]=p
                    break
                p,v=end,velocity
                pos[j,k+1]=p
                valid[j]=k+1
                if cursor<len(path) and np.linalg.norm(p-path[cursor])<.75:
                    cursor+=1
    return pos,valid


def target_samples(public,centers,spacing,contract,prior,seed):
    rng=np.random.default_rng(seed)
    prob=np.array(public['belief'],float)
    n=96
    ids=np.searchsorted(np.cumsum(prob/prob.sum()),(np.arange(n)+rng.random(n))/n)
    points=np.asarray(centers)[ids]
    boxes=tuple((tuple(p),tuple(spacing)) for p in np.asarray(centers)[np.asarray(public['occupied'],bool)])
    geometry=GeometryHypothesis(boxes)
    # Preserve the public cell probabilities. Any public belief mass on occupied cells
    # starts there and uses world-boundary reflection only for those particular draws.
    free=geometry.free(points,contract['target_obstacle_clearance'])
    direction=rng.normal(size=(n,3))
    direction/=np.maximum(np.linalg.norm(direction,axis=1)[:,None],1e-9)
    speed=rng.uniform(prior['speed_min'],prior['speed_max'],n)
    speed[rng.random(n)<prior['stationary_mass']]=0.
    vel=direction*speed[:,None]
    paths=[points.copy()]
    budget=Budget(20)
    for k in range(16):
        for mask,geom in [(free,geometry),(~free,GeometryHypothesis(()))]:
            if np.any(mask):
                points[mask],vel[mask]=propagate(points[mask],vel[mask],geom,public['t']+k,contract,budget)
        paths.append(points.copy())
    return np.stack(paths,axis=1),ids,int(sum(~free))


def team_first(boats,targets,radii):
    q=len(boats[0]['valid'])
    h=targets.shape[1]-1
    first=np.zeros((len(targets),q),int)
    for k in range(1,h+1):
        hit=np.zeros_like(first,dtype=bool)
        for boat,radius in zip(boats,radii):
            p=np.asarray(boat['positions'])
            distance=swept_distances(p[None,:,k-1],p[None,:,k],targets[:,None,k-1],targets[:,None,k])
            hit |= (distance<=radius)&(np.asarray(boat['valid'])[None,:]>=k)
        first[(first==0)&hit]=k
    return first
