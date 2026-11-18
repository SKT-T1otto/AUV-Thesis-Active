"""Use the same accepted view for D2 validity checks and objective scoring."""
from dataclasses import replace
from chapter3_bser.d2.controller import D2SearchController
from chapter3_bser.safety.public_geometry import PublicGeometry
from chapter3_bser.r123_task_sync.receipts import snapshot


class SyncSearchController(D2SearchController):
    def _geometry_pending(self,state):
        if not self.owner.task_sync_enabled or state.target_found:return super()._geometry_pending(state)
        if not self.owner.safe_options['path_safety']:return set()
        geometry=PublicGeometry(self.owner.planning_views.live(),self.owner.planning_views.clearance)
        receipts=snapshot(self.owner,state);pending=set(self.owner.safe_pending)
        for item in self.inner.current_allocation.search_assignments:
            r=receipts.get(item.agent_id)
            if r is None:
                remaining=self.owner.bridge.path_tracker.snapshot(item.agent_id).remaining_path_points
            else:
                remaining=r.points[r.route_index:] if r.status in ('ACTIVE','CONNECTOR') else r.points
            if not geometry.path_free((state.agents[item.agent_id].position,*remaining)):pending.add(item.agent_id)
        return pending

    def step(self,state,mission_context=None):
        if not self.owner.task_sync_enabled or state.target_found:return super().step(state,mission_context)
        prior=self.inner.current_allocation;receipts=snapshot(self.owner,state);items=[];views=[]
        self.owner.sync_pending_proposal=prior
        for item in prior.search_assignments:
            r=receipts.get(item.agent_id)
            if r is not None and r.status in ('ACTIVE','UNREACHABLE'):
                stored=self.owner.sync_templates.get(item.agent_id)
                if stored is None or stored[0]!=r.task_id:raise RuntimeError('accepted task template missing')
                template=stored[1]
                items.append(replace(template,waypoint=r.root,path=r.points,physical_travel_time=r.seconds))
                views.append(dict(agent=r.agent,task_id=r.task_id,proposal=list(item.waypoint),accepted=list(r.root),route_id=r.route_id))
            else:items.append(item)
        # Keep the original suggestion object in the journal; never force the
        # execution layer to accept it. Canonical planning now sees accepted IDs.
        self.owner.sync_planner_views.append(dict(step=state.step,agents=views))
        self.inner.current_allocation=replace(prior,search_assignments=tuple(items))
        return super().step(state,mission_context)
