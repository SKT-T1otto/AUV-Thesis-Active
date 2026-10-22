"""Fixed-resolution public scan memory; no hidden geometry or event archive.

Voxel solid support is an explicitly approximate v3 prior, not the continuous
all-ray constraint solver used by v2. No age-based decay is applied.
"""
from collections import Counter
from dataclasses import dataclass
import numpy as np
from .geometry_hypotheses import GeometryHypothesis
from .numerics import digest


@dataclass(frozen=True)
class VoxelEvidenceFrame:
    delta:np.ndarray
    observed:np.ndarray
    hits:np.ndarray
    origins:np.ndarray
    scan_batches:int
    ray_measurements:int

    @property
    def nbytes(self):return sum(a.nbytes for a in (self.delta,self.observed,self.hits,self.origins))


class PersistentVoxelMap:
    def __init__(self,config,contract):
        self.config=config;self.b=config['bounded'];self.bounds=np.asarray(contract['bounds'],float)
        self.sensor_range=float(contract['obstacle_sensor_range'])
        self.shape=tuple(self.b['grid_shape']);self.spacing=self.bounds/np.asarray(self.shape)
        self.logodds=np.zeros(self.shape,float);self.observed=np.zeros(self.shape,bool)
        self.hit_seen=np.zeros(self.shape,bool);self.revision=0;self.counts=Counter()
        self.last_origins=np.empty((0,3));self.last_dirty=np.empty((0,3),int)
        self._boxes=();self._occupied=np.zeros(self.shape,bool)

    def _indices(self,points):
        p=np.asarray(points,float).reshape(-1,3)
        return np.clip(np.floor(p/self.spacing).astype(int),0,np.asarray(self.shape)-1)

    def observe(self,scans,budget):
        free=np.zeros(self.shape,np.int32);occupied=np.zeros(self.shape,np.int32)
        origins_all=[]
        for scan in scans:
            budget.check();origins=np.asarray(scan.origins,float);directions=np.asarray(scan.directions,float)
            directions=directions/np.linalg.norm(directions,axis=1)[:,None]
            distances=np.asarray(scan.distances,float);hits=np.asarray(scan.hit_mask,bool)
            origins_all.extend(origins)
            for i,origin in enumerate(origins):
                for j,direction in enumerate(directions):
                    distance=max(0.,float(distances[i,j]));hit=bool(hits[i,j])
                    maximum=self.sensor_range
                    for axis in range(3):
                        if abs(direction[axis])>1e-12:
                            maximum=min(maximum,max(0.,((self.bounds[axis] if direction[axis]>0 else 0.)-origin[axis])/direction[axis]))
                    if distance>maximum+self.config['geometry']['scan_tolerance']:
                        raise ValueError('public_scan_range_exceeds_contract')
                    distance=min(distance,maximum)
                    limit=max(0.,distance-(.55*self.b['scan_sample_spacing'] if hit else 0.))
                    if limit>0:
                        samples=max(1,int(np.ceil(limit/self.b['scan_sample_spacing'])))
                        points=origin+np.linspace(0.,limit,samples+1)[:,None]*direction
                        points=points[np.all((points>=0)&(points<=self.bounds),axis=1)]
                        cells=np.unique(self._indices(points),axis=0)
                        if len(cells):free[tuple(cells.T)]+=1
                    if hit:
                        endpoint=origin+direction*distance
                        if np.all((endpoint>=0)&(endpoint<=self.bounds)):
                            cell=tuple(self._indices(endpoint)[0]);occupied[cell]+=1
        delta=self.b['free_logodds']*np.minimum(free,3)+self.b['occupied_logodds']*np.minimum(occupied,3)
        return VoxelEvidenceFrame(delta,(free>0)|(occupied>0),occupied>0,np.asarray(origins_all,float).reshape(-1,3),
            len(scans),sum(len(s.origins)*len(s.directions) for s in scans))

    def _apply(self,frame):
        self.observed|=frame.observed;self.hit_seen|=frame.hits;self.logodds+=frame.delta
        np.clip(self.logodds,-self.b['logodds_clip'],self.b['logodds_clip'],out=self.logodds)
        self.last_origins=frame.origins
        if len(self.last_origins):
            # A coarse cell containing a sensor cannot be declared a complete
            # solid. Preserve hit_seen as ambiguity evidence, rather than invent
            # a solid enclosing that public free point.
            indices=self._indices(self.last_origins);where=tuple(indices.T)
            self.logodds[where]=np.minimum(self.logodds[where],0.)

    def _finish(self,before,frame):
        after=self.classes();dirty=np.argwhere(before!=after);self.last_dirty=dirty
        if len(dirty):self.revision+=1
        self._occupied=after==1
        self.counts['scan_batches']+=frame.scan_batches;self.counts['ray_measurements']+=frame.ray_measurements
        self.counts['changed_cells']+=len(dirty)
        return dirty

    def ingest(self,scans,budget):
        frame=self.observe(scans,budget);before=self.classes();self._apply(frame)
        return self._finish(before,frame)

    def replace_window(self,frames,new_frame,budget):
        before=self.classes();self.logodds.fill(0.);self.observed.fill(False);self.hit_seen.fill(False)
        for frame in frames:
            budget.check();self._apply(frame)
        return self._finish(before,new_frame)

    def classes(self):
        probability=1/(1+np.exp(-self.logodds))
        out=np.zeros(self.shape,np.int8)
        out[probability<=self.b['free_threshold']]=-1
        out[probability>=self.b['occupied_threshold']]=1
        return out

    def boxes(self):
        # Exact union of occupied voxels, coalesced only along x. No free cell
        # is swallowed to meet the box cap. Exceeding capacity is visible.
        solids=self.classes()==1;boxes=[]
        for y in range(self.shape[1]):
            for z in range(self.shape[2]):
                ids=np.flatnonzero(solids[:,y,z])
                if not len(ids):continue
                starts=np.r_[0,np.flatnonzero(np.diff(ids)>1)+1];ends=np.r_[starts[1:],len(ids)]
                for left,right in zip(starts,ends):
                    lo=np.array([ids[left],y,z])*self.spacing
                    hi=np.array([ids[right-1]+1,y+1,z+1])*self.spacing
                    boxes.append((tuple((lo+hi)/2),tuple(hi-lo)))
        if len(boxes)>self.b['max_boxes']-self.config['geometry']['unknown_boxes']:
            raise ValueError('persistent_geometry_box_capacity')
        self._boxes=tuple(boxes)
        return self._boxes

    def unknown_box_allowed(self,box):
        center,size=map(np.asarray,box);lo=center-size/2;hi=center+size/2
        lower=self._indices(lo)[0];upper=self._indices(np.nextafter(hi,lo))[0]
        region=tuple(slice(int(a),int(b)+1) for a,b in zip(lower,upper))
        return bool(np.all(self.classes()[region]==0) and not np.any(self.observed[region]))

    def summary(self):
        classes=self.classes()
        return dict(revision=self.revision,grid_shape=self.shape,grid_cells=int(np.prod(self.shape)),
            spacing=self.spacing.tolist(),known_free_cells=int(np.sum(classes==-1)),
            occupied_cells=int(np.sum(classes==1)),observed_cells=int(self.observed.sum()),
            hit_evidence_cells=int(self.hit_seen.sum()),base_boxes=len(self._boxes),
            state_array_bytes=self.logodds.nbytes+self.observed.nbytes+self.hit_seen.nbytes,
            counts=dict(self.counts),resolution_approximation=True)


class LocalGeometryBank:
    def __init__(self,config,contract,rng,memory):
        self.config,self.contract,self.rng,self.memory=config,contract,rng,memory
        self.extras=[[] for _ in range(config['maps'])];self.previous_base=None;self.counts=Counter()
        self.attempts=0;self.rebuilds=0;self.support_boxes=0;self.distinct_maps=0;self.constraint_rays=0

    def repair(self,budget):
        base=self.memory.boxes();changed=base!=self.previous_base;maps=[]
        g=self.config['geometry'];bounds=np.asarray(self.contract['bounds'])
        for index,old in enumerate(self.extras):
            budget.check()
            kept=[box for box in old if self.memory.unknown_box_allowed(box)]
            self.counts['unknown_boxes_removed']+=len(old)-len(kept)
            changed|=len(kept)!=len(old)
            if index:
                for _ in range(g['unknown_boxes']-len(kept)):
                    for _ in range(self.config['geometry_attempts']):
                        budget.check();self.attempts+=1
                        size=self.rng.uniform(g['size_min'],g['size_max'],3)
                        if np.any(size>=bounds):continue
                        center=self.rng.uniform(size/2,bounds-size/2);box=(tuple(center),tuple(size))
                        if self.memory.unknown_box_allowed(box):kept.append(box);changed=True;break
            self.extras[index]=kept
            maps.append(GeometryHypothesis(base+tuple(kept)))
        self.previous_base=base;self.support_boxes=len(base);self.distinct_maps=len({m.identity for m in maps})
        self.counts['local_geometry_transactions']+=1
        if changed:self.counts['geometry_changes']+=1
        return tuple(maps),changed
