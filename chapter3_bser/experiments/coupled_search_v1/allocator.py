"""Bounded route search; D2 executor policy and physical safety stay in charge."""
from dataclasses import replace
import numpy as np
from core.mapping.planning_state import extract_planning_state
from chapter3_bser.d2.allocator import EffectAllocator
from chapter3_bser.d2.solver import best_standby
from chapter3_bser.objective import build_objective_context, response_diagnostics
from chapter3_bser.online.types import OnlineAllocation
from chapter3_bser.safety.public_geometry import PublicGeometry
from chapter3_bser.types import SearchCandidate
from .numerics import Budget, locked
from .types import RouteProposal
from .region_proposals import generate_routes, remaining_cost
from .execution_forecast import forecast_route
from .discovery import signature, compress, select_team, team_score


class CoupledSearchAllocator(EffectAllocator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.session=self.owner.map_research_session;self.research=self.session.config
        self.decision=None;self.failed_step=None;self.core_active=False
        self.scored_steps=set()

    def _fallback(self,state,error):
        self.core_active=False;self.failed_step=int(state.step);self.decision=None
        self.session.counts['d2_fallbacks']+=1
        row=dict(kind='coupled_fallback',step=int(state.step),reason=str(error),planner='d2_v1')
        self.audit.append(row);self.session.records.append(row)

    def _context(self,state):
        if self.failed_step==state.step:raise ValueError('core_failed_this_step')
        if self.decision is not None and self.decision['step']==state.step:
            self.decision['budget'].check();return self.decision
        if not self.session.valid or not self.session.belief.valid:
            raise ValueError('public_joint_filter_unavailable:'+str(self.session.belief.error or self.session.error))
        budget=Budget(self.research['planning_seconds'],queries=self.research['path_queries'])
        # Copy current PUBLIC arrays and graph without changing the original
        # provider's refresh schedule or grafting private beliefs into it.
        live=self.owner.planning_views.search(extract_planning_state(self.owner.env));budget.check()
        self.session.counts['decision_public_snapshots']+=1
        targets=self.session.belief.forecast(budget);scenarios=self.session.errors.scenarios(state.step)
        self.decision=dict(step=int(state.step),state=live,targets=targets,scenarios=scenarios,budget=budget,
            geometry=PublicGeometry(live,self.owner.planning_views.clearance),cache={})
        self.core_active=True
        return self.decision

    def _references(self,context,current):
        if current is None:return {}
        actual={} if self.session.context is None else {x.agent_id:x for x in self.session.context.agent_assignments}
        result={};state=context['state'];geometry=context['geometry']
        for assignment in current.search_assignments:
            i=assignment.agent_id;item=actual.get(i)
            c=self._frozen_search_candidate(assignment);index=0;blocked=False
            if item is not None and item.assignment_id==assignment.candidate_id:
                points=locked(item.planned_path).reshape(-1,3)
                c=replace(c,path_points=points,waypoint=tuple(item.final_waypoint),source='actual_installed_reference')
                index=self.session.tracker_indices.get(i,0);blocked=bool(item.hold_state or not item.reachable)
            route=RouteProposal(c,index,item is not None and item.assignment_id==assignment.candidate_id,blocked)
            if blocked:result[i]=replace(route,candidate=replace(c,planning_cost=0.,physical_travel_time=0.));continue
            if not geometry.path_free((state.agents[i].position,*c.path_points[index:],c.waypoint)):
                # Unsafe installed references cannot enter the optimization.
                result[i]=replace(route,blocked=True,candidate=replace(c,planning_cost=0.,physical_travel_time=0.))
                continue
            result[i]=remaining_cost(state,route)
        return result

    def _evaluate(self,context,routes):
        signatures=[];forecasts=[];costs=[];cfg=self.research
        for route in routes:
            c=route.candidate
            key=(route.agent_id,tuple(map(tuple,c.path_points)),tuple(c.waypoint),route.tracker_index,route.reference,route.blocked)
            cached=context['cache'].get(key)
            if cached is None:
                forecast=forecast_route(route,context['state'].agents[route.agent_id],self.session.vehicle_specs[route.agent_id],
                    self.session.control_contract,context['scenarios'],context['geometry'],context['budget'])
                sig=signature(context['targets'],forecast,
                    self.session.vehicle_specs[route.agent_id]['sensor_range']+self.session.control_contract['detect_eps_bias'],
                    dt=self.session.control_contract['dt'],beta=cfg['beta'])
                context['cache'][key]=(sig,forecast)
            else:sig,forecast=cached
            signatures.append(sig);forecasts.append(forecast);costs.append(c.planning_cost/cfg['cost_scale'])
        return signatures,forecasts,costs

    def _score(self,context,routes):
        sigs,forecasts,costs=self._evaluate(context,routes)
        score=team_score(sigs,context['targets'].weights,context['scenarios'].weights,costs,
                         rho=self.research['rho'],lambda_cost=self.research['lambda_cost'])
        return score,forecasts

    def _standby(self,state,selected,standbys,executor):
        original=build_objective_context(state,selected,standbys,self.config)
        y=best_standby(selected,standbys,original)
        return executor or self.execution.assign_standby(state,y),response_diagnostics(selected,y,original).conditional_reachable_response_time

    def _propose(self,state,current,affected,executor_affected,trigger):
        context=self._context(state);live=context['state'];cfg=self.research;budget=context['budget']
        references=self._references(context,current)
        routes=generate_routes(live,context['targets'],affected,references,cfg,context['geometry'],budget)
        frozen=tuple(r for i,r in sorted(references.items()) if i not in affected)
        routes=routes+frozen
        expected=set(affected)|set(references)
        if not expected or any(not any(r.agent_id==i for r in routes) for i in expected):
            raise ValueError('missing_public_safe_route')
        sigs,forecasts,costs=self._evaluate(context,routes)
        groups=[];compression={}
        for i in sorted(expected):
            indices=tuple(j for j,r in enumerate(routes) if r.agent_id==i)
            local=compress([sigs[j] for j in indices],context['targets'].weights,[costs[j] for j in indices],
                limit=cfg['representatives'] if i in affected else 1,epsilon=cfg['compression_epsilon'],
                lambda_cost=cfg['lambda_cost'],reference_indices=tuple(k for k,j in enumerate(indices) if routes[j].reference))
            groups.append(tuple(indices[k] for k in local.indices))
            compression[str(i)]=dict(raw=len(indices),representatives=len(local.indices),epsilon=local.epsilon,
                reached_error_target=local.reached_error_target,stopped_by=local.stopped_by)
        chosen,score,count=select_team(groups,sigs,context['targets'].weights,context['scenarios'].weights,costs,
            rho=cfg['rho'],lambda_cost=cfg['lambda_cost'],budget=budget)
        selected=tuple(routes[j].candidate for j in chosen)
        # Executor standby selection still uses the original D2 context.
        standbys,executor=self.standby_pool(state,current,movable=executor_affected) if current else self.standby_pool(state)
        if not standbys:raise ValueError('missing_original_standby')
        self.last_standbys=tuple(standbys);self.standby_may_change=executor is None
        executor,response=self._standby(state,selected,standbys,executor)
        budget.check()
        old={} if current is None else {a.agent_id:a for a in current.search_assignments}
        assignments=tuple(old[c.agent_id] if c.agent_id not in affected else self._search_assignment(c) for c in selected)
        allocation=OnlineAllocation(assignments,executor,score.objective,score.detection_probability,response,trigger,'COUPLED_FINITE_SEARCH')
        row=dict(kind='coupled_proposal',step=int(state.step),trigger=trigger,target_identity=context['targets'].identity,
            scenario_identity=context['scenarios'].identity,selected_ids=[c.candidate_id for c in selected],
            compression=compression,combinations=count,path_queries=budget.queries,elapsed_seconds=budget.elapsed,
            objective=score.objective,detection=score.detection_probability,scenario_utilities=score.scenario_values,
            scenario_detection=score.scenario_detection,frozen_agents=sorted(set(references)-set(affected)),installed=False)
        # Independent Monte Carlo resampling of this finite bank is diagnostic;
        # it is explicitly not independent ground-truth performance evidence.
        rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],int(state.step),3491]))
        sample=rng.choice(len(context['targets'].weights),cfg['review_samples'],p=context['targets'].weights)
        union=np.max([sigs[j].values for j in chosen],axis=0)
        row['finite_bank_review_mean_utility']=float(np.mean(union[sample]@context['scenarios'].weights))
        row['review_kind']='independent_resampling_of_same_finite_bank'
        self.audit.append(row);self.session.records.append(row);self.session.counts['core_proposals']+=1
        self.session.prediction=dict(step=int(state.step),installed_hash=None,
            paths={routes[j].agent_id:tuple(map(tuple,routes[j].candidate.path_points)) for j in chosen},
            finals={routes[j].agent_id:tuple(routes[j].candidate.waypoint) for j in chosen},
            forecasts={routes[j].agent_id:forecasts[j] for j in chosen})
        return allocation

    def allocate(self,state,*,trigger_reason='online'):
        if state.target_found:return super().allocate(state,trigger_reason=trigger_reason)
        current=getattr(getattr(self.owner,'controller',None),'current_allocation',None)
        try:return self._propose(state,current,set(state.searcher_ids),True,trigger_reason)
        except (ValueError,RuntimeError) as error:
            self._fallback(state,error);return super().allocate(state,trigger_reason=trigger_reason)

    def allocate_partial(self,state,current,*,affected_searcher_ids=(),executor_affected=False,trigger_reason):
        if state.target_found:return super().allocate_partial(state,current,affected_searcher_ids=affected_searcher_ids,
            executor_affected=executor_affected,trigger_reason=trigger_reason)
        if not set(affected_searcher_ids).issubset(state.searcher_ids):return current,False,'ATOMIC_REJECT_UNKNOWN_SEARCHER'
        try:
            return self._propose(state,current,set(affected_searcher_ids),executor_affected,trigger_reason),True,'COUPLED_ATOMIC_PROPOSAL'
        except (ValueError,RuntimeError) as error:
            self._fallback(state,error)
            return super().allocate_partial(state,current,affected_searcher_ids=affected_searcher_ids,
                executor_affected=executor_affected,trigger_reason=trigger_reason)

    def rescore(self,state,allocation,*,finalize=False):
        if state.target_found or allocation.search_frozen:return allocation
        try:
            context=self._context(state);routes=tuple(self._references(context,allocation).values())
            score,_=self._score(context,routes)
            if int(state.step) not in self.scored_steps:
                self.scored_steps.add(int(state.step));self.session.counts['core_scored_steps']+=1
                self.session.counts['last_core_scored_step']=int(state.step)
            if finalize and self.standby_may_change and self.last_standbys:
                executor,response=self._standby(state,tuple(r.candidate for r in routes),self.last_standbys,None)
                allocation=replace(allocation,executor_assignment=executor,response_time=response)
            return replace(allocation,objective_value=score.objective,detection_probability=score.detection_probability)
        except (ValueError,RuntimeError) as error:
            self._fallback(state,error);return super().rescore(state,allocation,finalize=finalize)
