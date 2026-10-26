"""Current shared public occupancy and endpoint-consistent local route queries."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from core.mapping.planning_state import extract_planning_state
from core.mapping.travel_cost_service import TravelCostService
from chapter3_bser.safety.public_geometry import PublicGeometry,filter_planning_state


class QueryView:
    def __init__(self,env,goal):self.env,self.goal=env,tuple(goal)
    def __getattr__(self,name):return getattr(self.env,name)
    def get_agent_state(self):
        state=self.env.get_agent_state();targets=list(state.navigation_targets)
        targets[3]=self.goal
        return replace(state,navigation_targets=tuple(targets))


def query_route(owner,goal):
    # A new read-only public snapshot has the current continuous start AND the
    # semantic standby goal as endpoints. Do not refresh/mutate D2's provider,
    # controller cache, allocation or its normal decision refresh schedule.
    state=extract_planning_state(QueryView(owner.env,goal))
    state=filter_planning_state(state,owner.planning_views.clearance)
    agent=next(a for a in state.agents if a.agent_id==state.executor_id)
    return TravelCostService(state).query(agent.position,goal,agent),int(state.step)


class ExecutorGeometry:
    def __init__(self,owner):
        live=owner.planning_views.live();planner=owner.env.unwrapped.map_module
        free=planner.known_free_mask.detach().cpu().numpy().astype(bool).reshape(-1).copy()
        occupied=np.asarray(live.occupancy.occupied_mask,bool).copy()
        state=SimpleNamespace(grid=live.grid,occupancy=SimpleNamespace(occupied_mask=occupied))
        self.live=live;self.margin=float(owner.planning_views.clearance)
        self.raw=PublicGeometry(state,0.)
        self.padded=PublicGeometry(state,self.margin+.1)
        self.free=free.reshape(live.grid.shape)
        self.origin=np.asarray(live.grid.origin,float)
        self.spacing=np.asarray(live.grid.spacing,float)
        self.shape=np.asarray(live.grid.shape,int)
        self.revision=int(getattr(planner,'map_revision',0))

    def clearance(self,point):
        p=np.asarray(point,float)
        if not len(self.raw.occupied_lower):return float('inf')
        delta=np.maximum(np.maximum(self.raw.occupied_lower-p,p-self.raw.occupied_upper),0.)
        return float(np.linalg.norm(delta,axis=1).min())

    def outward(self,point):
        p=np.asarray(point,float)
        if not len(self.raw.occupied_lower):return np.zeros(3)
        closest=np.clip(p,self.raw.occupied_lower,self.raw.occupied_upper)
        i=int(np.argmin(np.linalg.norm(p-closest,axis=1)));delta=p-closest[i]
        return delta/max(1e-12,float(np.linalg.norm(delta)))

    def segment(self,start,end,*,relaxed=False):
        geometry=self.raw if relaxed else self.padded
        return geometry.segment_free(start,end) and self.known_segment(start,end)

    def known_segment(self,start,end):
        # Exact closed-segment / unknown-voxel slab tests on the segment's small
        # cell bounding box. Avoid testing thousands of distant unknown voxels
        # on every predicted physical step. Adjacent boundary cells count too.
        p,q=np.asarray(start,float),np.asarray(end,float)
        if not self.raw.segment_free(p,q):return False
        lower=np.maximum(0,np.floor((np.minimum(p,q)-self.origin-1e-10)/self.spacing).astype(int))
        upper=np.minimum(self.shape-1,np.floor((np.maximum(p,q)-self.origin+1e-10)/self.spacing).astype(int))
        cells=np.array(list(np.ndindex(tuple(upper-lower+1))),int)+lower
        blocked=cells[~self.free[tuple(cells.T)]]
        if not len(blocked):return True
        lo=self.origin+blocked*self.spacing;hi=lo+self.spacing;delta=q-p
        near=np.full_like(lo,-np.inf);far=np.full_like(lo,np.inf)
        outside=np.zeros(len(lo),bool)
        for axis in range(3):
            if delta[axis]==0:outside|=(p[axis]<lo[:,axis])|(p[axis]>hi[:,axis])
            else:
                a,b=(lo[:,axis]-p[axis])/delta[axis],(hi[:,axis]-p[axis])/delta[axis]
                near[:,axis]=np.minimum(a,b);far[:,axis]=np.maximum(a,b)
        hits=(np.maximum(0.,near.max(axis=1))<=np.minimum(1.,far.min(axis=1)))&~outside
        return not bool(hits.any())
