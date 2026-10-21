"""Episode-local inference, public execution errors and installed-route audit."""
import numpy as np
from .audit import PublicEvidenceSession, installed_route_hash
from .coupled_belief import CoupledBelief
from .execution_forecast import PublicExecutionErrors


class CoupledSearchSession(PublicEvidenceSession):
    def __init__(self, config, vehicle_specs, control_contract):
        super().__init__(config,vehicle_specs,control_contract)
        self.belief=CoupledBelief(config,control_contract,vehicle_specs)
        self.errors=PublicExecutionErrors(config,vehicle_specs,control_contract)
        self.context=None;self.tracker_indices={};self.prediction=None

    def consume(self, events):
        evidence=super().consume(events)
        for item in evidence:
            if item.transition is not None:
                obs=item.transition.payload
                self.errors.observe(obs,self.context)
                if self.prediction is not None:
                    age=item.physical_step-self.prediction['step']
                    for agent_id,forecast in self.prediction['forecasts'].items():
                        if 0<age<forecast.positions.shape[1]:
                            error=np.linalg.norm(forecast.positions[:,age]-np.asarray(obs.agent_end[agent_id]),axis=1)
                            self.records.append(dict(kind='execution_prediction_error',step=item.physical_step,
                                agent_id=agent_id,age=age,scenario_position_errors=error.tolist(),
                                route_sha256=self.prediction['installed_hash']))
            self.belief.consume(item)
            terminal=bool(item.transition and (item.transition.payload.found_after or item.transition.payload.episode_terminated))
            if not terminal and not self.belief.stopped:
                self.counts['search_batches']+=1
                if self.belief.valid and self.belief.step==item.physical_step:
                    self.counts['available_search_batches']+=1
                else:
                    self.counts['unavailable_search_batches']+=1
                    raise RuntimeError('coupled_search_unavailable: step='+str(item.physical_step)+', reason='+str(self.belief.error))
        return evidence

    def installed(self, context, tracker_indices=None):
        route_hash=installed_route_hash(context)
        if self.prediction is not None:
            matches=True
            by_id={x.agent_id:x for x in context.agent_assignments}
            for agent_id,path in self.prediction['paths'].items():
                item=by_id.get(agent_id)
                if (item is None or item.hold_state or not item.reachable or tuple(map(tuple,item.planned_path))!=path
                        or tuple(item.final_waypoint)!=self.prediction['finals'][agent_id]):
                    matches=False;break
            expected=self.prediction['installed_hash']
            if not matches or (expected is not None and route_hash!=expected):
                self.records.append(dict(kind='forecast_invalidated',step=int(context.step),
                    reason='installed_path_hold_or_replacement',previous_route_sha256=expected,new_route_sha256=route_hash))
                self.counts['forecast_invalidations']+=1;self.prediction=None
            else:self.prediction['installed_hash']=route_hash
        super().installed(context)
        self.context=context;self.tracker_indices=dict(tracker_indices or {})

    def summary(self):
        result=super().summary()
        result.update(schema='ch3.coupled_search.core_audit.v1',mode='coupled_search',
            search_planner_id='coupled_search_v2',algorithm_changed=True,estimator_implemented=True,
            coupled_belief=self.belief.summary(),execution_error_observations=self.errors.observations,
            prior_is_approximate=True,performance_passed=False)
        return result
