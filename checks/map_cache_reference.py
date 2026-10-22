"""Uncached algorithm oracle for the cache-only change, not a runtime module.

The support/merge ordering is deliberately explicit. The independently guarded,
unchanged production entry kernel is used; no archived modules are executed.
"""
import numpy as np
from chapter3_bser.experiments.coupled_search_v1.geometry_hypotheses import GeometryHypothesis, RayConstraints


def compatible(geometry, scan, bounds, sensor_range, tolerance):
    origins=np.repeat(np.asarray(scan.origins,dtype=float),len(scan.directions),axis=0)
    direction=np.tile(np.asarray(scan.directions,dtype=float),(len(scan.origins),1))
    limits=np.full(len(origins),float(sensor_range));bounds=np.asarray(bounds,dtype=float)
    for axis in range(3):
        d=direction[:,axis];travel=np.full(len(d),np.inf)
        np.divide(np.where(d>0,bounds[axis],0.)-origins[:,axis],d,out=travel,where=abs(d)>1e-12)
        limits=np.minimum(limits,np.maximum(0.,travel))
    best=limits.copy()
    if not np.all(geometry.free(origins)):
        return False
    for center,size in geometry.boxes:
        lo=np.asarray(center)-np.asarray(size)/2;hi=np.asarray(center)+np.asarray(size)/2
        parallel=abs(direction)<=1e-12
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
    observed_hit=np.asarray(scan.hit_mask).reshape(-1);observed=np.asarray(scan.distances).reshape(-1)
    hit_ok=(predicted_hit==observed_hit)&(abs(best-observed)<=tolerance)
    free_ok=(~observed_hit)&(best>=observed-tolerance)
    return bool(np.all(hit_ok|free_ok))


def matches(rays, geometry):
    if not np.all(geometry.free(rays.origins)):
        return False
    best = rays.limits.copy()
    for box in geometry.boxes:
        best = np.minimum(best, rays.entry(box))
    predicted = best < rays.limits-1e-10
    return bool(np.all(((predicted == rays.hits) & (abs(best-rays.distances) <= rays.tolerance)) |
                       ((~rays.hits) & (best >= rays.distances-rays.tolerance))))


class ReferenceBank:
    def __init__(self, config, contract, rng):
        self.config, self.contract, self.rng = config, contract, rng
        self.kernel_calls = 0
        self.support_boxes = 0

    def propose(self, scans, budget):
        g = self.config['geometry']; bounds = np.asarray(self.contract['bounds'])
        rays = RayConstraints(scans, bounds, self.contract['obstacle_sensor_range'], g['scan_tolerance'], entry_cache_bytes=0)
        raw_entry = rays.entry
        def counted(box):
            self.kernel_calls += 1
            return raw_entry(box)
        rays.entry = counted
        boxes = []; best = rays.limits.copy()
        for index in np.flatnonzero(rays.hits):
            budget.check()
            if best[index] < rays.limits[index]-1e-10 and abs(best[index]-rays.distances[index]) <= rays.tolerance:
                continue
            point = rays.origins[index]+rays.directions[index]*rays.distances[index]
            half = min(.05, rays.tolerance/4); seed = None
            for _ in range(24):
                lo = np.maximum(0., point-half); hi = np.minimum(rays.bounds, point+half)
                trial = (tuple((lo+hi)/2), tuple(hi-lo))
                entry = rays.entry(trial)
                if (rays.allowed(entry) and np.all(GeometryHypothesis((trial,)).free(rays.origins))
                        and entry[index] < rays.limits[index]-1e-10 and abs(entry[index]-rays.distances[index]) <= rays.tolerance):
                    seed = trial; break
                half *= .5
            if seed is None:
                raise ValueError('public_ray_constraints_inconsistent')
            chosen = None
            if boxes:
                order = np.argsort([np.linalg.norm(np.asarray(b[0])-point) for b in boxes])[:g['merge_neighbors']]
                for j in order:
                    budget.check()
                    center, size = map(np.asarray, boxes[j]); left = center-size/2; right = center+size/2
                    if np.linalg.norm(np.maximum(0., np.maximum(left-point, point-right))) > g['cluster_radius']:
                        continue
                    lo = np.minimum(left, np.asarray(seed[0])-np.asarray(seed[1])/2)
                    hi = np.maximum(right, np.asarray(seed[0])+np.asarray(seed[1])/2)
                    merged = (tuple((lo+hi)/2), tuple(hi-lo))
                    if rays.allowed(rays.entry(merged)):
                        chosen = int(j); seed = merged; break
            if chosen is None:
                boxes.append(seed)
            else:
                boxes[chosen] = seed
            best = np.minimum(best, rays.entry(seed))
        base = GeometryHypothesis(tuple(boxes))
        if not matches(rays, base):
            raise ValueError('constructed_geometry_does_not_match_public_rays')
        accepted = [base]
        for _ in range(self.config['maps']-1):
            boxes = list(base.boxes)
            for _ in range(g['unknown_boxes']):
                for _ in range(self.config['geometry_attempts']):
                    budget.check()
                    size = self.rng.uniform(g['size_min'], g['size_max'], 3)
                    if np.any(size >= bounds):
                        continue
                    center = self.rng.uniform(size/2, bounds-size/2)
                    box = (tuple(center), tuple(size))
                    if rays.allowed(rays.entry(box)):
                        boxes.append(box); break
            candidate = GeometryHypothesis(tuple(boxes))
            if not matches(rays, candidate):
                raise ValueError('completion_violates_public_rays')
            accepted.append(candidate)
        self.support_boxes = len(base.boxes)
        return tuple(accepted)
