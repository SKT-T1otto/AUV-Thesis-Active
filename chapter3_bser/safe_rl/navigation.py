"""Scoped navigation repair; assignments never count as task completion."""
from collections import Counter
from dataclasses import replace
import numpy as np
from core.mapping.planning_state import extract_planning_state
from core.mapping.travel_cost_service import TravelCostService
from chapter3_bser.controllers.path_tracker import PathTracker
from chapter3_bser.safety.public_geometry import filter_planning_state
from chapter3_bser.safe_rl.evidence import EvidenceGeometry


class EndpointView:
    """Current continuous start and CURRENT semantic query goal, for any role."""
    def __init__(self,env,agent_id,goal):self.env=env;self.agent_id=agent_id;self.goal=tuple(goal)
    def __getattr__(self,name):return getattr(self.env,name)
    def get_agent_state(self):
        state=self.env.get_agent_state();targets=list(state.navigation_targets);targets[self.agent_id]=self.goal
        return replace(state,navigation_targets=tuple(targets))


class SafetyNavigationBridge:
    def __init__(self,inner,controller,env,options):
        self.inner=inner;self.controller=controller;self.env=env;self.options=options
        self.path_tracker=inner.path_tracker;self.repair_tracker=PathTracker(threshold=.25)
        self.routes={};self.last_query={};self.quarantine={};self.counts=Counter();self.trace=[]
        self.semantic_goals={}
        self.retry_pending=set();self.wait_anchors={};self.phases={}

    def __getattr__(self,name):return getattr(self.inner,name)

    def _route(self,i,goal,geometry):
        state=filter_planning_state(extract_planning_state(EndpointView(self.env,i,goal)),self.options['clearance'])
        agent=next(a for a in state.agents if a.agent_id==i)
        route=TravelCostService(state).query(agent.position,goal,agent)
        self.counts['route_queries']+=1
        if route.reachable and geometry.path(route.path_points):return tuple(tuple(float(x) for x in p) for p in route.path_points)
        return None

    def compile_guidance(self,allocation,planning_state,mission_context,*,decision_reason=None):
        context=self.inner.compile_guidance(allocation,planning_state,mission_context,decision_reason=decision_reason)
        state=extract_planning_state(self.env);step=int(state.step)
        self.controller.evidence.consume(self.env.read_public_events(self.controller.evidence.cursor))
        geometry=EvidenceGeometry(state,self.controller.evidence,self.options)
        self.semantic_goals={x.agent_id:x.final_waypoint for x in context.agent_assignments}
        finished=self.env.get_search_execution_state().agent_finished;output=[]
        for item in context.agent_assignments:
            i=item.agent_id;goal=tuple(item.final_waypoint);p=np.array(state.agents[i].position)
            phase=(bool(state.target_found),bool(item.execution_request))
            if finished[i] or self.phases.get(i,phase)!=phase:
                self.routes.pop(i,None);self.repair_tracker.reset(i);self.retry_pending.discard(i);self.wait_anchors.pop(i,None)
            self.phases[i]=phase
            if finished[i]:output.append(item);continue
            key=(i,goal);hazard=geometry.distance(goal)<=geometry.padding
            stored=self.routes.get(i)
            if stored is not None and stored['original_goal']!=goal:
                self.routes.pop(i,None);self.repair_tracker.reset(i);stored=None;self.retry_pending.discard(i);self.wait_anchors.pop(i,None)
            if stored is not None:
                arrived=np.linalg.norm(p-np.asarray(stored['goal']))<=self.repair_tracker.threshold
                expired=stored['goal']!=goal and step>=stored['created']+self.options['quarantine_steps']
                if (arrived and stored['goal']!=goal) or expired:
                    self.routes.pop(i,None);self.repair_tracker.reset(i);stored=None
                    self.retry_pending.add(i);self.counts['stage_arrivals' if arrived else 'stage_expirations']+=1
            need=hazard or self.controller.blocked[i]>=self.options['reject_after'] or i in self.retry_pending
            if need and step-self.last_query.get(i,-100000)>=self.options['repair_interval']:
                self.last_query[i]=step;self.counts['repair_attempts']+=1
                if hazard and self.quarantine.get(key,-1)<=step:
                    self.quarantine[key]=step+self.options['quarantine_steps'];self.counts['target_quarantines']+=1
                if not hazard:self.quarantine.pop(key,None)
                candidates=[] if self.quarantine.get(key,-1)>step else [goal]
                centers=np.asarray(state.grid.cell_centers)
                valid=np.asarray(state.occupancy.free_mask,bool)
                score=np.linalg.norm(centers-np.asarray(goal),axis=1)+.25*np.linalg.norm(centers-p,axis=1)
                for index in np.argsort(score):
                    candidate=tuple(centers[index])
                    if valid[index] and np.linalg.norm(centers[index]-p)>2*self.repair_tracker.threshold and geometry.distance(candidate)>geometry.padding+.15 and candidate not in candidates:
                        candidates.append(candidate)
                    if len(candidates)>=6:break
                replacement=None
                for candidate in candidates:
                    path=self._route(i,candidate,geometry)
                    if path is not None:replacement=dict(original_goal=goal,goal=tuple(candidate),path=path,created=step);break
                if replacement is not None:
                    self.routes[i]=replacement;self.repair_tracker.reset(i);stored=replacement;self.counts['route_replacements']+=1
                    self.retry_pending.discard(i);self.wait_anchors.pop(i,None)
                elif stored is not None and not geometry.path(stored['path']):
                    self.routes.pop(i,None);stored=None
                if replacement is None:self.counts['repair_unavailable']+=1;self.retry_pending.add(i)
            if stored is not None:
                tracking=self.repair_tracker.tracking_target(i,p,stored['path'],stored['goal'])
                if geometry.segment(p,tracking):
                    item=replace(item,final_waypoint=stored['goal'],planned_path=stored['path'],tracking_waypoint=tuple(tracking),
                        hold_position=tuple(p),hold_state=False,reachable=True)
                else:stored=None;self.routes.pop(i,None);self.retry_pending.add(i);need=True
            if stored is None and need:
                anchor=self.controller.anchors[i]
                anchor=self.wait_anchors.setdefault(i,p.copy() if anchor is None else np.asarray(anchor).copy())
                item=replace(item,tracking_waypoint=tuple(anchor),hold_position=tuple(anchor),hold_state=True,reachable=False)
            if need or stored is not None:
                self.trace.append(dict(step=step,agent_id=i,semantic_goal=list(goal),navigation_goal=list(item.final_waypoint),
                    tracking=list(item.tracking_waypoint),reason='observed_target_conflict' if hazard else 'repeated_unsafe_motion',
                    repaired=stored is not None,hold=item.hold_state,task_completion_claimed=False,map_revision=state.map_revision))
            output.append(item)
        executor=next(x for x in output if x.agent_id==3)
        return replace(context,agent_assignments=tuple(output),executor_assignment=replace(context.executor_assignment,
            tracking_waypoint=executor.tracking_waypoint,hold_position=executor.hold_position,
            hold_state=executor.hold_state,reachable=executor.reachable,planned_path=executor.planned_path))
