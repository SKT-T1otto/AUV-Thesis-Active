"""Physical public start connectors; never move virtual agents or edit the map."""
import numpy as np
from chapter3_bser.safety.planning_refresh import start_status
from chapter3_bser.path_option.path_context import Route


def connector(state, agent, geometry):
    if start_status(state,agent)=='ready':return None
    p=np.asarray(state.agents[agent].position)
    centers=np.asarray(state.grid.cell_centers)
    valid=np.asarray(state.planning_graph.valid_mask)&np.asarray(state.occupancy.free_mask)
    near=np.flatnonzero(valid & (np.linalg.norm(centers-p,axis=1)<=3.))
    for k in near[np.argsort(np.linalg.norm(centers[near]-p,axis=1))][:4]:
        q=centers[k]
        if np.linalg.norm(q-p)>.05 and geometry.segment(p,q):
            return Route([p,q],state.map_revision,state.step,True,
                         'public_start_connector',distance=float(np.linalg.norm(q-p)))
    return None


def independent_replans(inner,state,agents,install):
    """Original allocator/objective, one requested agent per atomic proposal."""
    records=[]
    for agent in sorted(agents):
        status=start_status(state,agent)
        if status!='ready':
            records.append(dict(agent=agent,accepted=False,reason=status));continue
        proposed,accepted,reason=inner.allocator.allocate_partial(state,inner.current_allocation,
            affected_searcher_ids=(agent,),executor_affected=False,trigger_reason='R123_REPAIR_RELEASE')
        if accepted:install(proposed,agent)
        records.append(dict(agent=agent,accepted=bool(accepted),reason=reason))
    return records
