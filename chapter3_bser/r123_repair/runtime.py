"""Minimal rule repair and explicit deletion control; reward/physics inherited."""
import copy
from dataclasses import replace
import numpy as np
from chapter3_bser.r123.runtime import PathOptionRuntime, observed_occupied
from chapter3_bser.r123.options import CONTROL
from chapter3_bser.r123.lifecycle import Task, arrived
from chapter3_bser.path_option.path_context import Route
from chapter3_bser.safe_rl.dynamics import limits
from chapter3_bser.reach_avoid.options import SPECS
from chapter3_bser.reach_avoid.scan_geometry import refined_buffer_exit
from chapter3_bser.safety.planning_refresh import start_status
from chapter3_bser.r123_repair.lifecycle import Lifecycle, ProgressManager
from chapter3_bser.r123_repair.planning import connector, independent_replans

VARIANTS=('legacy','repaired','no_recovery')


class RuleRuntime(PathOptionRuntime):
    def __init__(self,*args,variant='repaired',**kwargs):
        if variant not in ('repaired','no_recovery'):raise ValueError('invalid repaired rule variant')
        if kwargs.get('learner') is not None:raise ValueError('rule comparison cannot train or load a learner')
        super().__init__(*args,**kwargs)
        self.variant=variant
        self.lifecycle=Lifecycle(CONTROL);self.lifecycle_events=self.lifecycle.events
        self.manager=ProgressManager(self.o,self.lifecycle,True)
        self._task_tokens={};self._connectors={};self._last_root_query={}

    def assignment(self,i):
        allocation=self.controller.inner.current_allocation
        if i<3:
            item=next((x for x in allocation.search_assignments if x.agent_id==i),None)
            if item is None or item.failure_reason is not None:
                return dict(assigned=False,token=None,point=None,kind='UNASSIGNED')
            return dict(assigned=True,token=(str(item.candidate_id),tuple(item.waypoint)),
                        point=np.asarray(item.waypoint),kind='search')
        item=allocation.executor_assignment
        return dict(assigned=True,token=(str(item.source),tuple(item.target_region)),
                    point=np.asarray(item.target_region),kind='executor')

    def _original(self,i,step):
        a=self.assignment(i)
        return a['point'].copy() if a['assigned'] else super()._original(i,step)

    def _clear(self,i):
        self._alternatives.pop(i,None);self._last_candidates.pop(i,None)
        self._connectors.pop(i,None)

    def _one(self,i,c,original,reason):
        candidates=[(c.endpoint.copy(),c.route)]*8
        self._candidates[i]=candidates
        self._has_choices=getattr(self,'_has_choices',{});self._has_choices[i]=False
        mask=np.zeros(8,bool);mask[0]=True
        return c,original,reason,candidates,mask

    def _idle(self,i,phase,now,p,event):
        self.lifecycle.attempts.pop(i,None)
        self.lifecycle.tasks[i]=Task(0,p.copy(),phase,now,now,p.copy(),p.copy(),released=True,release_reason=event)
        self.lifecycle.events.append(dict(agent=i,step=now,event=event,phase=phase))
        c=self.manager.commit(i,p,p,p,now,phase,Route([p],0,now,True),reason=event)
        self._clear(i)
        return c

    def _wait(self,i,state,packet,original):
        c=self.manager.current[i]
        self._release_requests.add(i);self._request_refresh=True
        planning=self.planning_views.search(state)
        # Let the inherited *verified* safety exit act even when no task exists.
        # A coarse-planner connector is physical motion, not a fictitious graph start.
        if i<3 and start_status(planning,i)!='ready':
            cached=self._connectors.get(i)
            if cached is None or state.step-cached.query_step>=self.o['route_query_interval']:
                cached=connector(planning,i,self._geometry)
                self._connectors[i]=cached
                self.lifecycle.events.append(dict(agent=i,step=state.step,event='CONNECTOR_QUERY',
                    reachable=cached is not None,start_status=start_status(planning,i)))
            if cached is not None and self._geometry.segment(packet.positions[i],cached.points[-1]):
                c=self.manager.commit(i,original,c.task,cached.points[-1],state.step,int(packet.phases[i]),
                    copy.deepcopy(cached),reason='PUBLIC_START_CONNECTOR') if c.mode!='CONNECTOR' or np.linalg.norm(c.endpoint-cached.points[-1])>.05 else c
                c.mode='CONNECTOR'
                c.route.advance(packet.positions[i],.15)
                return self._one(i,c,original,'CONNECTOR')
        return self._one(i,c,original,'WAIT_TASK')

    def _options(self,i,state,packet):
        p=packet.positions[i];phase=int(packet.phases[i]);now=state.step
        original=self._original(i,now);a=self.assignment(i)
        self.manager.public_pose[i]=(bool(packet.pose_free[i]),float(packet.clearances[i]))
        task=self.lifecycle.tasks.get(i);c=self.manager.current.get(i)
        if phase==3:
            if task is None or task.phase!=3:c=self._idle(i,phase,now,p,'PHASE_FROZEN')
            self._release_requests.discard(i)
            return self._one(i,c,original,'FROZEN')
        if task is None or task.phase!=phase:
            if not a['assigned']:
                c=self._idle(i,phase,now,p,'UNASSIGNED')
            else:
                point=packet.intents[i] if phase==2 else original
                self.lifecycle.accept(i,point,phase,now,p)
                self._task_tokens[i]=a['token']
                c=self.manager.commit(i,original,point,point,now,phase,self._query(i,point,state),reason='PHASE_OR_INITIAL')
                self._clear(i)
            task=self.lifecycle.tasks[i]
        if not a['assigned']:
            if task.identity and not task.released:self.lifecycle.release(i,now,'UNASSIGNED')
            return self._wait(i,state,packet,original)
        if task.released:
            point=packet.intents[i] if phase==2 else original
            changed=a['token']!=self._task_tokens.get(i)
            same_resume=task.identity>0 and task.release_reason not in ('TASK_COMPLETE','UNASSIGNED')
            # Occupying a hold coordinate is not a new search assignment.
            already_done=arrived(p,packet.velocities[i],point,phase,bool(packet.pose_free[i]),float(packet.clearances[i]),packet.target_velocity)
            if (changed or same_resume or phase==2) and (phase!=0 or not already_done):
                route=self._query(i,point,state)
                if route.reachable and (phase==2 or start_status(self.planning_views.search(state),i)=='ready'):
                    if phase!=2 and (changed or not task.identity):
                        self.lifecycle.accept(i,point,phase,now,p);task=self.lifecycle.tasks[i]
                    else:
                        task.released=False;task.root_only=True
                    self._task_tokens[i]=a['token']
                    c=self.manager.commit(i,original,point,point,now,phase,route,reason='D2_RESUME')
                    self._clear(i);self._release_requests.discard(i)
                else:return self._wait(i,state,packet,original)
            else:return self._wait(i,state,packet,original)
        c,raw,reason,candidates,mask=self._legacy_options(i,state,packet)
        task=self.lifecycle.tasks[i]
        if phase==2:task.point=c.task.copy()
        done=arrived(p,packet.velocities[i],c.endpoint,phase,bool(packet.pose_free[i]),float(packet.clearances[i]),packet.target_velocity)
        if c.mode=='RECOVER' and done:
            route=self._query(i,c.task,state)
            if route.reachable:
                self.lifecycle.events.append(dict(agent=i,step=now,event='RECOVERY_RETURN',task_id=task.identity))
                task.progress_step=now;task.recovery_start=None;self.lifecycle.attempts.pop(i,None)
                self.manager.commit(i,raw,c.task,c.task,now,phase,route,reason='RECOVERY_RETURN')
                self._clear(i)
                return self._legacy_options(i,state,packet)
            reason='NO_ROUTE'
        if c.mode=='TASK' and done and phase==0:
            self.lifecycle.release(i,now,'TASK_COMPLETE')
            return self._wait(i,state,packet,raw)
        if c.mode=='RECOVER' and task.recovery_start is not None and now-task.recovery_start>=CONTROL['recovery_budget_steps']:
            reason='RECOVERY_EXPIRED'
        if task.root_only and c.mode=='TASK' and c.route.reachable and reason=='STALLED':
            # Root tracking after exhausted recovery must not inherit the same
            # release/resume loop. Keep the budget exhausted and requery only.
            if now-self._last_root_query.get(i,-10000)>=self.o['route_query_interval']:
                c.route=self._query(i,c.task,state);self._last_root_query[i]=now
                c.route.advance(p,self.o['arrival_radius'])
            if c.route.reachable:return self._one(i,c,raw,'TRACK_ROOT')
            reason='NO_ROUTE'
        failed=reason in ('STALLED','NO_ROUTE','KNOWN_INVALID','RECOVERY_EXPIRED')
        if failed and (self.variant=='no_recovery' or task.root_only or task.recoveries>=CONTROL['max_recoveries'] or reason=='RECOVERY_EXPIRED'):
            self.lifecycle.release(i,now,'RECOVERY_EXHAUSTED' if self.variant=='repaired' else 'NO_ACTIVE_RECOVERY')
            return self._wait(i,state,packet,raw)
        if failed:
            mask=mask.copy()
            for j,(point,_) in enumerate(candidates):
                if np.linalg.norm(point-c.endpoint)<.5:mask[j]=False
            self._has_choices[i]=bool(mask.any())
            if not mask.any():
                self.lifecycle.release(i,now,'NO_RECOVERY_ROUTE')
                return self._wait(i,state,packet,raw)
        if self.variant=='no_recovery' or task.root_only:
            return self._one(i,c,raw,'TRACK_ROOT' if reason=='STALLED' else reason)
        return c,raw,reason,candidates,mask

    def _legacy_options(self,i,state,packet):
        if self.variant!='no_recovery':return super()._legacy_options(i,state,packet)
        # No alternative endpoint generation/query/selection in deletion control.
        c=self.manager.current[i];p=packet.positions[i];phase=int(packet.phases[i]);now=state.step
        if phase==2 and np.linalg.norm(packet.intents[i]-c.task)>=.75:
            c.task=packet.intents[i].copy();c.endpoint=c.task.copy();c.route=self._query(i,c.task,state)
        c.route.advance(p,self.o['arrival_radius'])
        if now-c.route.query_step>=self.o['route_query_interval'] and (not c.route.reachable or observed_occupied(state,c.route.tracking(p))):
            c.route=self._query(i,c.endpoint,state);c.route.advance(p,self.o['arrival_radius'])
        reason=self.manager.status(i,p,packet.velocities[i],phase,now,not observed_occupied(state,c.endpoint),packet.target_velocity if phase==2 else None)
        return self._one(i,c,self._original(i,now),reason)

    def observe(self):
        fresh=self._cached is None or self._cached[0]!=self.step
        rows=super().observe()
        if not fresh:return rows
        for i,row in enumerate(rows):
            if row['phase']==3 or row['reason'] in ('WAIT_TASK','CONNECTOR','TRACK_ROOT'):
                row.update(rule_terminate=False,forced_terminate=False,beta_allowed=False)
            if row['reason']=='WAIT_TASK':
                result=refined_buffer_exit(self._packet.positions[i],self._packet.velocities[i],SPECS[i],
                    self._geometry,self.guard.contract,self.guard.options,self._packet.flow[i])
                if result is not None:
                    row['actions'][:,9]=result[0]/limits(SPECS[i])
                    row['action_mask'][:]=False;row['action_mask'][:,9]=True;row['forced'][:]=9
            a=self.assignment(i)
            row['assignment_kind']=a['kind'];row['assignment_token']=a['token']
        return rows

    def _refresh_tasks(self):
        if self.terminal or not (self._request_refresh or self._release_requests):return
        if self.step-self._last_request<CONTROL['request_interval']:return
        from chapter3_bser.learning.base_runtime import _public_context
        from chapter3_bser.online.types import BSERActionAssignment
        from chapter3_bser.events.event_types import BSEREvent
        self.state=self.provider.snapshot(force=True);context=_public_context(self.env,self.state)
        result=self.controller.step(self.state,context);inner=self.controller.inner
        def install(proposed,agent):
            nonlocal result
            old=inner.current_allocation
            updates=inner.waypoints.updates(old,proposed,reason='R123_REPAIR_RELEASE',step=self.step)
            inner.current_allocation=proposed;inner.waypoints.commit(proposed)
            inner.policy.mark_replan(self.step,BSEREvent.WAYPOINT_STALE)
            inner.replan_count+=1;inner.replan_steps.append(self.step)
            result=BSERActionAssignment(self.step,True,result.events,proposed,updates,'R123_REPAIR_RELEASE',result.event_detection)
        affected=[i for i in self._release_requests if i<3]
        if affected and not self.state.target_found:
            planning=self.planning_views.search(self.state)
            for record in independent_replans(inner,planning,affected,install):
                self.lifecycle.events.append(dict(step=self.step,event='D2_RELEASE_REQUEST',**record))
        self.env.observe_controller_result(result,controller=self.controller,state_provider=self.provider)
        self.guidance=self.bridge.compile_guidance(result.allocation,self.state,context,decision_reason=result.decision_reason)
        self._cached=None;self._alternatives.clear();self._last_candidates.clear()
        self._last_request=self.step;self._request_refresh=False

    def _actions_for_step(self,actions):
        before={i:c.mode for i,c in self.manager.current.items()}
        commands=super()._actions_for_step(actions)
        for i,(_,decision) in enumerate(self._pending):
            if before.get(i)=='RECOVER' and decision['terminate'] and self.manager.current[i].mode=='TASK':
                # Selecting the root after a failed detour is a safe fallback,
                # not an arrival and not permission to immediately detour again.
                self.lifecycle.tasks[i].root_only=True
                self.lifecycle.attempts.pop(i,None)
                self.lifecycle.events.append(dict(agent=i,step=self.step,event='RECOVERY_ROOT_FALLBACK',
                    task_id=self.lifecycle.task_id(i)))
        return commands

    def advance(self,*args,**kwargs):
        result=super().advance(*args,**kwargs)
        for row,(state,_) in zip(self.trace[-4:],self._pending):
            row.update(assignment_kind=state['assignment_kind'],assignment_token=state['assignment_token'])
        return result

    def summary(self):
        result=super().summary()
        result['method']='auv.r123.rule_repair.v1'
        result['path_option']['rule_variant']=self.variant
        return result
