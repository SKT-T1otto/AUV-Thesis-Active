"""Heterogeneous prior-control rollout and public-error execution scenarios."""
from collections import deque
from dataclasses import dataclass
import numpy as np
from .numerics import digest, locked
from .types import ExecutionForecast


def nominal_step(position, velocity, target, spec, contract, *, error=(0.,0.,0.)):
    """Original zero-residual equations; error is a synthetic velocity increment."""
    p=np.asarray(position,float);v=np.asarray(velocity,float).copy();relative=np.asarray(target)-p
    acceleration=np.zeros(3);dt=contract['dt'];eps=contract['eps']
    if contract['use_residual_prior']:
        distance=max(eps,float(np.linalg.norm(relative[:2])))
        desired=relative[:2]/distance*spec['v_xy_max']*np.clip(distance/contract['prior_slow_radius_xy'],0,1)
        acceleration[:2]=np.clip(contract['prior_kv_xy']*(desired-v[:2]),-spec['a_xy_max'],spec['a_xy_max'])
        desired_z=np.clip(relative[2]/contract['prior_slow_radius_z'],-1,1)*spec['v_z_max']
        acceleration[2]=np.clip(contract['prior_kv_z']*(desired_z-v[2]),-spec['a_z_max'],spec['a_z_max'])
        acceleration*=contract['prior_strength_search']
        acceleration[:2]=np.clip(acceleration[:2],-spec['a_xy_max'],spec['a_xy_max'])
        acceleration[2]=np.clip(acceleration[2],-spec['a_z_max'],spec['a_z_max'])
    v+=acceleration*dt
    v[:2]+=(-spec['drag_xy']*v[:2])*dt
    v[2]+=(-spec['drag_z']*v[2]+spec['buoyancy_bias'])*dt
    v+=np.asarray(error)
    v[:2]*=min(1.,spec['v_xy_max']/max(eps,float(np.linalg.norm(v[:2]))))
    v[2]=np.clip(v[2],-spec['v_z_max'],spec['v_z_max'])
    raw=p+v*dt;result=np.clip(raw,0,contract['bounds'])
    if np.any(raw!=result):v*=.4
    return result,v


@dataclass(frozen=True)
class ExecutionScenarios:
    errors: np.ndarray  # Q,H,3 searchers,3 axes; joint temporal blocks
    weights: np.ndarray
    identity: str


class PublicExecutionErrors:
    def __init__(self, config, specs, contract):
        self.config,self.specs,self.contract=config,specs,contract
        self.history=deque(maxlen=config['execution_history']);self.observations=0

    def observe(self, transition, installed):
        if installed is None or not transition.target_transition_completed:return
        error=np.zeros((3,3))
        by_id={item.agent_id:item for item in installed.agent_assignments}
        for i in range(3):
            item=by_id.get(i)
            if item is None or item.hold_state or not item.reachable:continue
            _,prediction=nominal_step(transition.agent_start[i],transition.velocity_start[i],
                item.tracking_waypoint,self.specs[i],self.contract)
            error[i]=np.asarray(transition.velocity_end[i])-prediction
        self.history.append(np.clip(error,-self.config['execution_error_clip'],self.config['execution_error_clip']))
        self.observations+=1

    def scenarios(self, step):
        q,h=self.config['execution_scenarios'],self.config['horizon']
        errors=np.zeros((q,h,3,3))
        # Each scenario keeps the observed cross-agent vector and contiguous time
        # order. It does not draw independent perturbations per route or step.
        if self.history:
            history=np.asarray(self.history);length=len(history)
            for e in range(1,q):
                start=(e-1)*max(1,length//max(1,q-1))
                errors[e]=history[(start+np.arange(h))%length]
        identity=digest(dict(step=int(step),errors=errors.tolist(),observations=self.observations))
        return ExecutionScenarios(locked(errors),locked(np.full(q,1/q)),identity)


def forecast_route(proposal, agent, spec, contract, scenarios, geometry, budget, *, threshold=.75):
    candidate=proposal.candidate;path=np.asarray(candidate.path_points,float).reshape(-1,3)
    final=np.asarray(candidate.waypoint,float);q,h=scenarios.errors.shape[:2]
    positions=np.empty((q,h+1,3));valid=np.zeros(q,dtype=int);reasons=[]
    if proposal.blocked:
        positions[:]=np.asarray(agent.position)
        return ExecutionForecast(locked(positions),locked(valid,dtype=int),('installed_hold',)*q,scenarios.identity)
    for e in range(q):
        p=np.asarray(agent.position,float);v=np.asarray(agent.velocity,float);index=proposal.tracker_index
        positions[e,0]=p;reason='horizon'
        if not proposal.reference and index<len(path) and np.linalg.norm(p-path[index])<threshold:index+=1
        for k in range(h):
            budget.check()
            target=path[index] if index<len(path) else final
            end,new_v=nominal_step(p,v,target,spec,contract,error=scenarios.errors[e,k,agent.agent_id])
            if not geometry.segment_free(p,end):
                reason='public_safety_truncation';positions[e,k+1:]=p;break
            p,v=end,new_v;positions[e,k+1]=p;valid[e]=k+1
            if index<len(path) and np.linalg.norm(p-path[index])<threshold:index+=1
        reasons.append(reason)
    return ExecutionForecast(locked(positions),locked(valid,dtype=int),tuple(reasons),scenarios.identity)
