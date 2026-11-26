"""Public interpretation of a pre-dispatch forecast; a routing veto cannot certify absence of sensing."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class OfferEvidence:
    admission: str
    route_support: str
    first_scan_cells: np.ndarray

    def conditional_bounds(self, forecast_probability, valid_steps, horizon):
        """Bounds conditional on the first native target scan actually taking place.

        We preserve the frozen cumulative forecast when all execution samples cover
        the horizon. With truncated samples we expose forecast hits through their valid
        prefixes as the lower estimate and leave the unmodeled continuation open.
        A checked native physical segment includes its starting point, regardless of motion.
        """
        lower=np.array(forecast_probability,float,copy=True)
        upper=lower.copy() if np.all(np.asarray(valid_steps)>=horizon) else np.ones_like(lower)
        lower[self.first_scan_cells]=1.
        upper[self.first_scan_cells]=1.
        return lower,upper


def interpret_offer(centers,position,radius,goal,tokens,prior,*,reachable,blocked):
    """Use only a current public receipt and this offer's route check (never a future ACK)."""
    admission='needs_native_receipt'
    if prior and prior['status']=='RELEASED' and prior['token']:
        if prior['token'][0] in tokens and np.allclose(prior['root'],goal,atol=1e-5,rtol=0):
            admission='released_token_offered_again'
    support='no_public_route' if not reachable else 'public_geometry_veto' if blocked else 'route_available'
    inside=np.linalg.norm(np.asarray(centers)-position,axis=1)<=radius
    return OfferEvidence(admission,support,np.flatnonzero(inside))


def predict_public_offer(state, agent_id, goal, query, *, tokens, prior_receipt, history_scenarios,
                         calibration, specs, contract, clearance, arrival_radius, sensor_radius):
    """Public entry point. Return the frozen conditional preview and its dispatch evidence."""
    from chapter3_bser.predispatch_sweep import preview_sweep

    prior_route=None
    if prior_receipt and prior_receipt['status']=='ACTIVE' and prior_receipt['reachable'] and prior_receipt['token']:
        if (prior_receipt['token'][0] in tokens and tuple(prior_receipt['root'])==tuple(goal)
                and tuple(prior_receipt['endpoint'])==tuple(goal)):
            prior_route=prior_receipt['points'],prior_receipt['route_index']
    preview=preview_sweep(state,agent_id,goal,query,history_scenarios=history_scenarios,calibration=calibration,
                  specs=specs,contract=contract,clearance=clearance,arrival_radius=arrival_radius,
                  sensor_radius=sensor_radius,prior_route=prior_route)
    evidence=interpret_offer(state.grid.cell_centers,state.agents[agent_id].position,sensor_radius,goal,tokens,
                       prior_receipt,reachable=query.reachable,blocked=preview.blocked if preview else False)
    return preview,evidence
