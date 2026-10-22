"""Public-ray-constrained AABB supports and unobserved completions (revision 2).

Observed surfaces are constructed, never accepted by blind rejection sampling.
Every support and merge respects all retained free-ray intervals. This remains
an approximate conditional geometry prior, not a reconstruction of hidden truth.
"""
from dataclasses import dataclass
import numpy as np
from .numerics import digest
from .geometry_cache import ByteLRU, ObservationCache, box_key, prepare_scan, readonly


@dataclass(frozen=True)
class GeometryHypothesis:
    boxes: tuple  # ((center xyz), (size xyz)), spatially coherent solids

    @property
    def identity(self):
        return digest(self.boxes)

    def obstacles(self):
        return tuple(dict(center=center,size=size) for center,size in self.boxes)

    def free(self, points, clearance=0.0):
        p=np.asarray(points,dtype=float)
        result=np.ones(p.shape[:-1],dtype=bool)
        for center,size in self.boxes:
            lo=np.asarray(center)-np.asarray(size)/2-clearance
            hi=np.asarray(center)+np.asarray(size)/2+clearance
            result &= ~np.all((p>lo)&(p<hi),axis=-1)
        return result


def compatible(geometry, scan, bounds, sensor_range, tolerance, *, prepared=None):
    if prepared is None:
        prepared=prepare_scan(scan,bounds,sensor_range)
    origins=prepared.origins;direction=prepared.directions;limits=prepared.limits
    best=limits.copy()
    if not np.all(geometry.free(origins)):
        return False
    for center,size in geometry.boxes:
        lo=np.asarray(center)-np.asarray(size)/2
        hi=np.asarray(center)+np.asarray(size)/2
        parallel=prepared.parallel
        outside=np.any(parallel&((origins<lo)|(origins>hi)),axis=1)
        first=np.zeros_like(origins);second=np.zeros_like(origins)
        np.divide(lo-origins,direction,out=first,where=~parallel)
        np.divide(hi-origins,direction,out=second,where=~parallel)
        near=np.where(parallel,-np.inf,np.minimum(first,second))
        far=np.where(parallel,np.inf,np.maximum(first,second))
        enter=np.maximum(0.,near.max(axis=1));leave=far.min(axis=1)
        hit=(~outside)&(enter<=leave)&(leave>=0)&(enter<best)
        best[hit]=enter[hit]
    predicted_hit=best<limits-1e-10
    observed_hit=prepared.hits
    observed=prepared.distances
    # Tolerance is declared geometric approximation, not a fake miss rate.
    hit_ok=(predicted_hit==observed_hit)&(abs(best-observed)<=tolerance)
    free_ok=(~observed_hit)&(best>=observed-tolerance)
    return bool(np.all(hit_ok|free_ok))


class GeometryHypothesisBank:
    def __init__(self, config, contract, rng, *, scan_cache=None):
        self.config,self.contract,self.rng=config,contract,rng
        self.attempts=0
        self.support_boxes=0
        self.distinct_maps=0
        self.constraint_rays=0
        self.rebuilds=0
        self.scan_cache=scan_cache if scan_cache is not None else ObservationCache(config['history_steps']+1)
        self.last_entry_stats={}

    def _supports(self, rays, budget):
        """Cover all hits; the neighbor budget limits merge work, not coverage."""
        boxes=[];best=rays.limits.copy();g=self.config['geometry']
        for index in np.flatnonzero(rays.hits):
            budget.check()
            if best[index]<rays.limits[index]-1e-10 and abs(best[index]-rays.distances[index])<=rays.tolerance:
                continue
            point=rays.origins[index]+rays.directions[index]*rays.distances[index]
            # Shrink only the invented thickness, never discard a measured hit.
            half=min(.05,rays.tolerance/4)
            seed=None
            for _ in range(24):
                lo=np.maximum(0.,point-half);hi=np.minimum(rays.bounds,point+half)
                trial=(tuple((lo+hi)/2),tuple(hi-lo))
                entry=rays.evaluate_entry(trial)
                if (rays.allowed(entry) and np.all(GeometryHypothesis((trial,)).free(rays.origins))
                        and entry[index]<rays.limits[index]-1e-10 and abs(entry[index]-rays.distances[index])<=rays.tolerance):
                    seed=trial;seed_entry=entry;break
                half*=.5
            if seed is None:raise ValueError('public_ray_constraints_inconsistent')
            chosen=None
            if boxes:
                order=np.argsort([np.linalg.norm(np.asarray(b[0])-point) for b in boxes])[:g['merge_neighbors']]
                for j in order:
                    budget.check()
                    center,size=map(np.asarray,boxes[j]);left=center-size/2;right=center+size/2
                    if np.linalg.norm(np.maximum(0.,np.maximum(left-point,point-right)))>g['cluster_radius']:
                        continue
                    lo=np.minimum(left,np.asarray(seed[0])-np.asarray(seed[1])/2)
                    hi=np.maximum(right,np.asarray(seed[0])+np.asarray(seed[1])/2)
                    merged=(tuple((lo+hi)/2),tuple(hi-lo))
                    merged_entry=rays.evaluate_entry(merged)
                    if rays.allowed(merged_entry):
                        chosen=int(j);seed=merged;seed_entry=merged_entry;break
            if chosen is None:boxes.append(seed)
            else:
                rays.forget_entry(boxes[chosen]);boxes[chosen]=seed
            rays.remember_entry(seed,seed_entry)
            best=np.minimum(best,seed_entry)
        geometry=GeometryHypothesis(tuple(boxes))
        if not rays.matches(geometry):raise ValueError('constructed_geometry_does_not_match_public_rays')
        return geometry

    def propose(self, scans, budget):
        if not scans:raise ValueError('public_scan_history_required')
        g=self.config['geometry'];bounds=np.asarray(self.contract['bounds'])
        rays=RayConstraints(scans,bounds,self.contract['obstacle_sensor_range'],g['scan_tolerance'],scan_cache=self.scan_cache)
        self.last_entry_stats=rays.entry_stats
        base=self._supports(rays,budget);accepted=[base]
        # The observed-support world is an explicit prior component. Optional
        # unseen solids add diversity only where every public ray permits them.
        for _ in range(self.config['maps']-1):
            boxes=list(base.boxes)
            for _ in range(g['unknown_boxes']):
                for _ in range(self.config['geometry_attempts']):
                    budget.check();self.attempts+=1
                    size=self.rng.uniform(g['size_min'],g['size_max'],3)
                    if np.any(size>=bounds):continue
                    center=self.rng.uniform(size/2,bounds-size/2)
                    box=(tuple(center),tuple(size))
                    entry=rays.evaluate_entry(box)
                    if rays.allowed(entry):
                        boxes.append(box);rays.remember_entry(box,entry);break
            candidate=GeometryHypothesis(tuple(boxes))
            if not rays.matches(candidate):raise ValueError('completion_violates_public_rays')
            accepted.append(candidate)
        self.support_boxes=len(base.boxes);self.distinct_maps=len({m.identity for m in accepted})
        self.constraint_rays=len(rays.origins);self.rebuilds+=1
        return tuple(accepted)


class RayConstraints:
    """Vectorized retained public observations; no environment or true layout."""
    def __init__(self,scans,bounds,sensor_range,tolerance,*,scan_cache=None,entry_cache_bytes=16*1024**2):
        self.bounds=np.asarray(bounds,float);self.tolerance=float(tolerance)
        values=[]
        for scan in scans:
            if scan_cache is not None:
                values.append(scan_cache.prepare(scan,bounds,sensor_range).rows)
                continue
            origins=np.repeat(np.asarray(scan.origins,float),len(scan.directions),axis=0)
            directions=np.tile(np.asarray(scan.directions,float),(len(scan.origins),1))
            values.append(np.column_stack((origins,directions,np.asarray(scan.distances).reshape(-1),np.asarray(scan.hit_mask).reshape(-1))))
        # Exact duplicate rays carry the same constraint; retain every distinct
        # observation, including contradictory measurements at the same origin.
        data=np.unique(np.concatenate(values),axis=0)
        self.origins=data[:,:3];self.directions=data[:,3:6]
        self.distances=data[:,6];self.hits=data[:,7].astype(bool)
        self.limits=np.full(len(data),float(sensor_range))
        for axis in range(3):
            d=self.directions[:,axis];travel=np.full(len(data),np.inf)
            np.divide(np.where(d>0,self.bounds[axis],0)-self.origins[:,axis],d,out=travel,where=abs(d)>1e-12)
            self.limits=np.minimum(self.limits,np.maximum(0.,travel))
        self.parallel=abs(self.directions)<=1e-12
        # A memo belongs to this exact, immutable row-sorted history snapshot.
        for name in ('bounds','origins','directions','distances','hits','limits','parallel'):
            setattr(self,name,readonly(getattr(self,name)))
        self._entries=ByteLRU(entry_cache_bytes,4096)
        self.entry_stats=dict(kernel_calls=0,cache_hits=0,retained_bytes=0,peak_bytes=0)

    def evaluate_entry(self,box):
        self.entry_stats['kernel_calls']+=1
        return self.entry(box)

    def remember_entry(self,box,entry):
        self._entries.put(box_key(box),readonly(entry),entry.nbytes)
        self.entry_stats['retained_bytes']=self._entries.bytes
        self.entry_stats['peak_bytes']=self._entries.peak_bytes

    def forget_entry(self,box):
        self._entries.discard(box_key(box))
        self.entry_stats['retained_bytes']=self._entries.bytes

    def cached_entry(self,box):
        entry=self._entries.get(box_key(box))
        if entry is not None:
            self.entry_stats['cache_hits']+=1
            return entry
        entry=self.evaluate_entry(box)
        self.remember_entry(box,entry)
        return entry

    def entry(self,box):
        center,size=map(np.asarray,box);lo=center-size/2;hi=center+size/2
        outside=np.any(self.parallel&((self.origins<lo)|(self.origins>hi)),axis=1)
        first=np.zeros_like(self.origins);second=np.zeros_like(self.origins)
        np.divide(lo-self.origins,self.directions,out=first,where=~self.parallel)
        np.divide(hi-self.origins,self.directions,out=second,where=~self.parallel)
        near=np.where(self.parallel,-np.inf,np.minimum(first,second))
        far=np.where(self.parallel,np.inf,np.maximum(first,second))
        enter=np.maximum(0.,near.max(axis=1));leave=far.min(axis=1)
        return np.where((~outside)&(enter<=leave)&(leave>=0)&(enter<self.limits),enter,np.inf)

    def allowed(self,entry):
        return bool(np.all(entry>=self.distances-self.tolerance))

    def matches(self,geometry):
        if not np.all(geometry.free(self.origins)):return False
        best=self.limits.copy()
        for box in geometry.boxes:best=np.minimum(best,self.cached_entry(box))
        predicted=best<self.limits-1e-10
        return bool(np.all(((predicted==self.hits)&(abs(best-self.distances)<=self.tolerance))|
                           ((~self.hits)&(best>=self.distances-self.tolerance))))
