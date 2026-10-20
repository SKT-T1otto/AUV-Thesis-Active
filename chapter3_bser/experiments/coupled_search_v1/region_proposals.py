"""Future marginals propose regions only; route utility uses first-hit samples."""
from dataclasses import replace
import numpy as np
from chapter3_bser.types import SearchCandidate
from core.mapping.travel_cost_service import TravelCostService
from .numerics import digest, locked
from .types import RouteProposal


def region_indices(state, targets, limit):
    centers=np.asarray(state.grid.cell_centers);valid=np.asarray(state.planning_graph.valid_mask)
    mass=np.zeros(len(centers))
    for k in range(1,targets.positions.shape[1]):
        points=targets.positions[:,k]
        nearest=np.argmin(np.sum((points[:,None,:]-centers[None,:,:])**2,axis=2),axis=1)
        np.add.at(mass,nearest,targets.weights/(targets.positions.shape[1]-1))
    # Frontier cells enter the proposal pool but do not add discovery utility.
    frontier=np.asarray(state.occupancy.unknown_mask,dtype=float)
    score=mass+.05*frontier/max(1,float(frontier.sum()))
    pool=list(np.flatnonzero(valid));selected=[]
    spacing=max(1e-9,float(np.linalg.norm(np.asarray(state.grid.spacing))))
    while pool and len(selected)<limit:
        if selected:
            separation=np.min(np.linalg.norm(centers[pool,None]-centers[selected][None,:],axis=2),axis=1)
            value=score[pool]*np.minimum(1.,separation/(2*spacing))
            # Spatial ties are resolved by distance from the current team center.
            value+=1e-12*separation
        else:
            team=np.mean([a.position for a in state.agents[:3]],axis=0)
            value=score[pool]+1e-12*np.linalg.norm(centers[pool]-team,axis=1)
        chosen=int(np.argmax(value));selected.append(pool.pop(chosen))
    return tuple(selected)


def _candidate(agent_id, points, cells, cost, time, source):
    points=locked(points);cells=locked(cells,dtype=int)
    route_id='coupled_'+digest(dict(agent=agent_id,path=points.tolist()))[:20]
    return SearchCandidate(agent_id,route_id,tuple(points[-1]),points,cells,
        float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),float(cost),float(time),source)


def generate_routes(state, targets, ids, references, config, geometry, budget):
    service=TravelCostService(state);regions=region_indices(state,targets,config['proposal_regions'])
    centers=np.asarray(state.grid.cell_centers);result=[]
    for agent_id in sorted(ids):
        agent=state.agents[agent_id];pool=[]
        reference=references.get(agent_id)
        if reference is not None and not reference.blocked:pool.append(reference)
        # Different agents rotate region order. Candidate IDs depend on full paths.
        order=regions[agent_id%max(1,len(regions)):]+regions[:agent_id%max(1,len(regions))]
        for index in order:
            if len(pool)>=max(1,config['raw_routes']*2//3):break
            budget.query();query=service.query(agent.position,centers[index],agent)
            if not query.reachable or not geometry.path_free(query.path_points):continue
            candidate=_candidate(agent_id,query.path_points,query.path_cell_indices,query.planning_cost,
                                 query.physical_travel_time,'coupled_future_region')
            if any(np.array_equal(candidate.path_points,r.candidate.path_points) for r in pool):continue
            pool.append(RouteProposal(candidate))
        # Via-region alternatives allow different discovery traces to a destination.
        direct=tuple(r for r in pool if not r.reference)
        for route_index,route in enumerate(direct):
            if len(pool)>=config['raw_routes'] or len(direct)<2:break
            other=direct[(route_index+1)%len(direct)]
            middle=route.candidate.waypoint;goal=other.candidate.waypoint
            budget.query();second=service.query(middle,goal,agent)
            if not second.reachable:continue
            points=np.vstack((route.candidate.path_points,second.path_points[1:]))
            if not geometry.path_free(points):continue
            cells=np.concatenate((route.candidate.path_cell_indices,second.path_cell_indices))
            c=_candidate(agent_id,points,cells,route.candidate.planning_cost+second.planning_cost,
                         route.candidate.physical_travel_time+second.physical_travel_time,'coupled_via_region')
            if not any(np.array_equal(c.path_points,r.candidate.path_points) for r in pool):pool.append(RouteProposal(c))
        result.extend(pool)
    return tuple(result)


def remaining_cost(state, proposal):
    """Original edge and continuous endpoint costs on the installed remainder.

    Public endpoint sets retain only the cheapest connector per component.
    A moving agent may still be tracking a different, already installed segment.
    Price that safe continuous segment by the original endpoint time equation;
    this neither adds a planning-graph connector nor changes the installed path.
    """
    c=proposal.candidate;agent=state.agents[c.agent_id]
    path=list(np.asarray(c.path_points)[proposal.tracker_index:])
    if not path or not np.array_equal(path[-1],c.waypoint):path.append(np.asarray(c.waypoint))
    points=[np.asarray(agent.position),*path];service=TravelCostService(state)
    cost=time=0.
    for start,end in zip(points,points[1:]):
        if np.linalg.norm(start-end)<1e-12:continue
        left=service._center_connector(start);right=service._center_connector(end)
        if left is not None and right is not None:
            edges=state.planning_graph.adjacency_for_role(agent.role)[left.cell_index]
            edge=next((e for e in edges if e.destination==right.cell_index),None)
            if edge is None:raise ValueError('reference_edge_unavailable')
            cost+=edge.planning_cost;time+=edge.physical_travel_time
        else:
            horizontal,vertical=state.planning_graph.time_rates_for_role(agent.role)
            delta=np.asarray(end)-np.asarray(start)
            segment=float(np.linalg.norm(delta[:2])*horizontal+abs(delta[2])*vertical)
            cost+=segment;time+=segment
    return replace(proposal,candidate=replace(c,planning_cost=float(cost),physical_travel_time=float(time)))
