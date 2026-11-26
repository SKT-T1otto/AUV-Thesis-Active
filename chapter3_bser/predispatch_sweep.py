"""First-sweep forecasts conditional on a proposed public goal being executed."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from chapter3_bser.execution_calibration import calibrated_route
from chapter3_bser.experiments.coupled_search_v1.execution_forecast import forecast_route
from chapter3_bser.experiments.coupled_search_v1.numerics import Budget
from chapter3_bser.experiments.coupled_search_v1.types import RouteProposal
from chapter3_bser.path_option.path_context import Route
from chapter3_bser.safety.public_geometry import PublicGeometry
from chapter3_bser.sweep_events import first_steps


@dataclass
class SweepPreview:
    path: np.ndarray
    cursor: int
    route_id: str
    blocked: bool
    positions: dict
    valid_steps: dict
    first: dict
    weights: np.ndarray

    def probability(self, model, horizon):
        times=self.first[model]
        return self.weights @ ((times>0)&(times<=horizon))


def preview_sweep(state, agent_id, goal, query, *, history_scenarios, calibration, specs, contract,
                  clearance, arrival_radius, sensor_radius, prior_route=None):
    """An unreachable query is not a predicted stationary scan. Return no map for it.

    prior_route is optional (points, cursor) from a still-active identical command.
    Only the caller's pre-solve public data are consumed; no environment or receipts
    from the pending command enter this function.
    """
    if not query.reachable:
        return None
    agent=state.agents[agent_id]
    points,index=prior_route if prior_route is not None else (query.path_points,0)
    route=Route(points,state.map_revision,state.step,True,index=int(index))
    route.advance(np.asarray(agent.position),arrival_radius)
    geometry=PublicGeometry(state,clearance)
    blocked=not geometry.path_free((agent.position,*route.points[route.index:],goal))
    candidate=SimpleNamespace(agent_id=agent_id,path_points=route.points,waypoint=goal)
    proposal=RouteProposal(candidate,route.index,True,blocked)
    nominal=forecast_route(proposal,agent,specs[agent_id],contract,history_scenarios,geometry,
                           Budget(20),threshold=arrival_radius)
    item=dict(id=agent_id,pos=agent.position,vel=agent.velocity,path=route.points,
              endpoint=np.asarray(goal),cursor=route.index)
    positions,valid=calibrated_route(calibration,item,specs[agent_id],contract,
            np.asarray(state.occupancy.occupied_mask).reshape(state.grid.shape),
            np.asarray(state.grid.spacing),geometry,blocked)
    paths={'C':nominal.positions,'new':positions}
    lengths={'C':nominal.valid_steps,'new':valid}
    weights=np.asarray(history_scenarios.weights)
    assert len(positions)==len(weights) and np.allclose(weights,1/len(weights))
    first={k:first_steps(state.grid.cell_centers,v,lengths[k],sensor_radius) for k,v in paths.items()}
    return SweepPreview(route.points,route.index,route.route_id,blocked,paths,lengths,first,weights)
