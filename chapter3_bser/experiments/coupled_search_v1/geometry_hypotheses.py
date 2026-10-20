"""Persistent continuous AABB completions constrained only by public scans.

Scan-anchored proposals are an explicitly approximate conditional completion
prior, not an exact reconstruction of the unknown geometry posterior.
"""
from dataclasses import dataclass
import numpy as np
from .numerics import digest


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


def compatible(geometry, scan, bounds, sensor_range, tolerance):
    origins=np.repeat(np.asarray(scan.origins,dtype=float),len(scan.directions),axis=0)
    direction=np.tile(np.asarray(scan.directions,dtype=float),(len(scan.origins),1))
    limits=np.full(len(origins),float(sensor_range))
    bounds=np.asarray(bounds,dtype=float)
    for axis in range(3):
        d=direction[:,axis]
        travel=np.full(len(d),np.inf)
        np.divide(np.where(d>0,bounds[axis],0.)-origins[:,axis],d,out=travel,where=abs(d)>1e-12)
        limits=np.minimum(limits,np.maximum(0.,travel))
    best=limits.copy()
    if not np.all(geometry.free(origins)):
        return False
    for center,size in geometry.boxes:
        lo=np.asarray(center)-np.asarray(size)/2
        hi=np.asarray(center)+np.asarray(size)/2
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
    observed_hit=np.asarray(scan.hit_mask).reshape(-1)
    observed=np.asarray(scan.distances).reshape(-1)
    # Tolerance is declared geometric approximation, not a fake miss rate.
    hit_ok=(predicted_hit==observed_hit)&(abs(best-observed)<=tolerance)
    free_ok=(~observed_hit)&(best>=observed-tolerance)
    return bool(np.all(hit_ok|free_ok))


class GeometryHypothesisBank:
    def __init__(self, config, contract, rng):
        self.config,self.contract,self.rng=config,contract,rng
        self.attempts=0
        self.uncovered_hit_clusters=0

    def _anchors(self, scans):
        anchors=[];radius=self.config['geometry']['cluster_radius']
        for scan in scans:
            for i,j in np.argwhere(np.asarray(scan.hit_mask)):
                d=np.asarray(scan.directions[j]);p=np.asarray(scan.origins[i])+d*scan.distances[i][j]
                if not any(np.linalg.norm(p-q)<radius for q,_ in anchors):
                    anchors.append((p,d))
        limit=self.config['geometry']['max_hit_clusters']
        self.uncovered_hit_clusters=max(0,len(anchors)-limit)
        return anchors[:limit]

    def propose(self, scans, budget):
        g=self.config['geometry'];bounds=np.asarray(self.contract['bounds'])
        anchors=self._anchors(scans)
        if self.uncovered_hit_clusters:
            raise ValueError('geometry_hit_cluster_budget')
        accepted=[]
        for attempt in range(self.config['geometry_attempts']):
            budget.check();self.attempts+=1
            boxes=[]
            for point,direction in anchors:
                size=self.rng.uniform(g['size_min'],g['size_max'],3)
                axis=int(np.argmax(abs(direction)));sign=1 if direction[axis]>0 else -1
                center=point+self.rng.uniform(-.35,.35,3)*size
                center[axis]=point[axis]+sign*size[axis]/2
                if np.any(center-size/2<0) or np.any(center+size/2>bounds):
                    break
                boxes.append((tuple(center),tuple(size)))
            else:
                for _ in range(g['unknown_boxes']):
                    size=self.rng.uniform(g['size_min'],g['size_max'],3)
                    if np.any(size>=bounds):
                        continue
                    center=self.rng.uniform(size/2,bounds-size/2)
                    boxes.append((tuple(center),tuple(size)))
                geometry=GeometryHypothesis(tuple(boxes))
                if all(compatible(geometry,s,bounds,self.contract['obstacle_sensor_range'],g['scan_tolerance']) for s in scans):
                    accepted.append(geometry)
                    if len(accepted)==self.config['maps']:
                        return tuple(accepted)
        raise ValueError('geometry_proposal_budget_exhausted')
