"""First-discovery signatures on shared weighted target/scenario samples."""
import itertools
from dataclasses import dataclass
import numpy as np
from .numerics import locked, probabilities, swept_distances
from .types import DiscoverySignature, CompressionResult


def signature(targets, execution, radius, *, dt, beta):
    t=np.asarray(targets.positions);a=np.asarray(execution.positions)
    if t.shape[1]!=a.shape[1]:raise ValueError('different signature horizons')
    first=np.zeros((len(t),len(a)),dtype=int)
    for k in range(1,t.shape[1]):
        distance=swept_distances(a[None,:,k-1],a[None,:,k],t[:,None,k-1],t[:,None,k])
        hit=(first==0)&(k<=np.asarray(execution.valid_steps)[None,:])&(distance<=radius)
        first[hit]=k
    values=np.where(first>0,np.exp(-beta*dt*first),0.)
    return DiscoverySignature(locked(values),locked(first,dtype=int),targets.identity,execution.identity)


def _shared(signatures, weights):
    if not signatures:raise ValueError('empty signature collection')
    key=(signatures[0].target_identity,signatures[0].execution_identity,signatures[0].values.shape)
    if any((s.target_identity,s.execution_identity,s.values.shape)!=key for s in signatures):
        raise ValueError('signatures must share target and execution scenarios')
    w=probabilities(weights)
    if len(w)!=key[2][0]:raise ValueError('target weight shape mismatch')
    return w


def signature_distance(a,b,weights,cost_a,cost_b,lambda_cost):
    w=_shared((a,b),weights)
    return float(np.max(np.sum(w[:,None]*np.abs(a.values-b.values),axis=0))
                 +lambda_cost*abs(cost_a-cost_b))


def compress(signatures, weights, costs, *, limit, epsilon, lambda_cost, reference_indices=()):
    _shared(signatures,weights)
    if len(costs)!=len(signatures) or limit<1:raise ValueError('invalid compression pool')
    selected=list(dict.fromkeys(reference_indices))
    if any(i<0 or i>=len(signatures) for i in selected) or len(selected)>limit:
        raise ValueError('invalid mandatory representatives')
    if not selected:selected=[0]
    distance=np.full(len(signatures),np.inf)
    for i in selected:
        distance=np.minimum(distance,[signature_distance(s,signatures[i],weights,c,costs[i],lambda_cost)
                                      for s,c in zip(signatures,costs)])
    while max(distance)>epsilon and len(selected)<min(limit,len(signatures)):
        index=int(np.argmax(distance));selected.append(index)
        distance=np.minimum(distance,[signature_distance(s,signatures[index],weights,c,costs[index],lambda_cost)
                                      for s,c in zip(signatures,costs)])
    error=float(max(distance));reached=error<=epsilon
    return CompressionResult(tuple(selected),error,reached,'error_target' if reached else 'representative_cap')


@dataclass(frozen=True)
class TeamScore:
    objective: float
    detection_probability: float
    scenario_values: tuple
    scenario_detection: tuple


def team_score(signatures, weights, scenario_weights, costs, *, rho, lambda_cost):
    w=_shared(signatures,weights);nu=probabilities(scenario_weights)
    if len(nu)!=signatures[0].values.shape[1] or len(costs)!=len(signatures):
        raise ValueError('team objective shape mismatch')
    union=np.max([s.values for s in signatures],axis=0)
    found=np.any([s.first_steps>0 for s in signatures],axis=0)
    values=np.sum(w[:,None]*union,axis=0);detection=np.sum(w[:,None]*found,axis=0)
    objective=(1-rho)*float(nu@values)+rho*float(min(values))-lambda_cost*sum(costs)
    return TeamScore(objective,float(nu@detection),tuple(values),tuple(detection))


def select_team(groups, signatures, weights, scenario_weights, costs, *, rho, lambda_cost, budget):
    """Exact product over compressed groups; frozen agents have one option."""
    if not groups or any(not group for group in groups):raise ValueError('missing search route group')
    best=None;best_score=None;count=0
    for indices in itertools.product(*groups):
        budget.check();count+=1
        score=team_score([signatures[i] for i in indices],weights,scenario_weights,[costs[i] for i in indices],
                         rho=rho,lambda_cost=lambda_cost)
        if best_score is None or score.objective>best_score.objective+1e-12:
            best,best_score=indices,score
    return best,best_score,count
