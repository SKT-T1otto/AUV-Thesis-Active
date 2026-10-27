"""Separate causal cache of EXISTING public scans and current occupancy views."""
from collections import OrderedDict
import numpy as np
from chapter3_bser.safety.public_geometry import PublicGeometry


class ScanEvidence:
    def __init__(self,capacity=8192):
        self.cursor=0;self.hits=OrderedDict();self.capacity=capacity
        self.last_scan_step=-1;self.scans=0;self.evictions=0;self.latest=None

    def consume(self,events):
        for event in events:
            if event.sequence<=self.cursor:raise ValueError('scan evidence cursor regressed')
            self.cursor=event.sequence
            if event.kind!='scan':continue
            self.latest=event.payload;self.last_scan_step=event.physical_step;self.scans+=1
            scan=event.payload;directions=np.asarray(scan.directions,float)
            for origin,distances,hits in zip(scan.origins,scan.distances,scan.hit_mask):
                for direction,distance,hit in zip(directions,distances,hits):
                    if not hit:continue
                    point=np.asarray(origin,float)+direction*distance
                    key=tuple(np.rint(point/.10).astype(int))
                    self.hits[key]=(point.copy(),int(event.physical_step))
                    self.hits.move_to_end(key)
                    if len(self.hits)>self.capacity:self.hits.popitem(last=False);self.evictions+=1

    def points(self):
        return np.array([v[0] for v in self.hits.values()],float).reshape(-1,3)


class EvidenceGeometry:
    """Known occupied voxels + retained scan hits. Absence is NOT certified free."""
    def __init__(self,state,evidence,options):
        self.state=state;self.raw=PublicGeometry(state,0.)
        self.padded=PublicGeometry(state,options['clearance'])
        self.points=evidence.points();self.padding=options['clearance']+options['hit_padding']
        self.origin=np.asarray(state.grid.origin,float);self.spacing=np.asarray(state.grid.spacing,float)
        self.shape=np.asarray(state.grid.shape,int)
        self.free=np.asarray(state.occupancy.free_mask,bool).reshape(tuple(self.shape))
        self.unknown=np.asarray(state.occupancy.unknown_mask,bool).reshape(tuple(self.shape))

    def point_distances(self,point):
        p=np.asarray(point,float)
        hit=float(np.linalg.norm(self.points-p,axis=1).min()) if len(self.points) else float('inf')
        if len(self.raw.occupied_lower):
            delta=np.maximum(np.maximum(self.raw.occupied_lower-p,p-self.raw.occupied_upper),0.)
            voxel=float(np.linalg.norm(delta,axis=1).min())
        else:voxel=float('inf')
        return hit,voxel

    def distance(self,point):return min(self.point_distances(point))

    def segment(self,start,end,extra=0.):
        if not self.padded.segment_free(start,end):return False
        p,q=np.asarray(start,float),np.asarray(end,float)
        # Include growing model-error padding against occupied cells, too.
        if extra>0 and len(self.raw.occupied_lower):
            delta=q-p;lo=self.raw.occupied_lower-self.padding-extra;hi=self.raw.occupied_upper+self.padding+extra
            near=np.full_like(lo,-np.inf);far=np.full_like(lo,np.inf);outside=np.zeros(len(lo),bool)
            for axis in range(3):
                if delta[axis]==0:outside|=(p[axis]<lo[:,axis])|(p[axis]>hi[:,axis])
                else:
                    a,b=(lo[:,axis]-p[axis])/delta[axis],(hi[:,axis]-p[axis])/delta[axis]
                    near[:,axis]=np.minimum(a,b);far[:,axis]=np.maximum(a,b)
            if np.any((np.maximum(0.,near.max(1))<=np.minimum(1.,far.min(1)))&~outside):return False
        if len(self.points):
            d=q-p;squared=float(np.dot(d,d))
            t=np.zeros(len(self.points)) if squared==0 else np.clip((self.points-p)@d/squared,0.,1.)
            gap=np.linalg.norm(self.points-(p+t[:,None]*d),axis=1)
            if np.any(gap<=self.padding+extra):return False
        return True

    def path(self,points):
        return bool(len(points)) and all(self.segment(a,b) for a,b in zip(points,points[1:])) and self.segment(points[-1],points[-1])

    def local(self,point):
        index=np.floor((np.asarray(point)-self.origin)/self.spacing).astype(int)
        index=np.clip(index,0,self.shape-1)
        slices=tuple(slice(max(0,int(i)-1),min(int(n),int(i)+2)) for i,n in zip(index,self.shape))
        return float(self.unknown[slices].mean()),bool(self.free[tuple(index)])

    def outward(self,point):
        p=np.asarray(point,float);vectors=[]
        if len(self.points):vectors.extend(p-self.points[np.argsort(np.linalg.norm(self.points-p,axis=1))[:4]])
        if len(self.raw.occupied_lower):
            closest=np.clip(p,self.raw.occupied_lower,self.raw.occupied_upper)
            vectors.append(p-closest[np.argmin(np.linalg.norm(p-closest,axis=1))])
        if not vectors:return np.zeros(3)
        vectors=np.asarray(vectors);norm=np.linalg.norm(vectors,axis=1)
        return (vectors/np.maximum(norm[:,None],1e-9)**2).sum(0)
