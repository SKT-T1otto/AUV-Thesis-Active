"""Existing no-recovery lifecycle with governed action construction and trace."""
import numpy as np
from core.mapping.planning_state import extract_planning_state
from chapter3_bser.r123_governed.runtime import RuleRuntime as ParentRuntime
from chapter3_bser.reach_avoid.scan_geometry import ScanRefinedGeometry,refined_buffer_exit
from chapter3_bser.reach_avoid.options import SPECS
from chapter3_bser.safe_rl.dynamics import limits
from chapter3_bser.r123_tracking.control import conditional_banks
from chapter3_bser.r123_tracking.progress import carry_progress,tracking_velocity


ARMS=dict(baseline=(False,False),path_only=(True,False),feedforward_only=(False,True),both=(True,True))


class RuleRuntime(ParentRuntime):
    def __init__(self,*args,tracking_arm="both",**kwargs):
        if tracking_arm not in ARMS:raise ValueError("unknown tracking arm")
        self.tracking_arm=tracking_arm
        self.tracking_switches=dict(zip(("path","feedforward"),ARMS[tracking_arm]))
        self.route_events=[]
        if kwargs.get('variant','no_recovery')!='no_recovery':raise ValueError('active recovery remains disabled')
        kwargs['variant']='no_recovery'
        super().__init__(*args,**kwargs)

    def observe(self):
        if self._cached is not None and self._cached[0] == self.step:
            return self._cached[1]
        state = extract_planning_state(self.env)
        events = self.env.read_public_events(self.guard.evidence.cursor)
        self.guard.observe(state, events); self.subcells.consume(events, state.step)
        geometry = ScanRefinedGeometry(state, self.guard.evidence, self.guard.options, self.subcells)
        self._geometry = geometry
        original = {i: self._original(i, state.step) for i in range(4)}
        packet = self.encoder.build(state, self.env.get_task_state(), self.observations, original, geometry,
                                    self.guard, self.env.get_public_delivered_target())
        self._packet = packet
        self._public_state = state
        rows = []
        for i in range(4):
            c, raw, reason, candidates, goal_mask = self._options(i, state, packet)
            p = packet.positions[i]; bounds = packet.bounds
            tracks = [r.tracking(p) for _, r in candidates]
            actions, action_mask, forced, emergency, bank_diagnostics = conditional_banks(
                packet, i, tracks, geometry, self.guard, self.switches['r1'],
                target_velocities=[tracking_velocity(packet,i,route,endpoint,self.tracking_switches['feedforward']) for endpoint,route in candidates])
            phase = int(packet.phases[i]); age = state.step-c.received
            if phase == 3:
                actions = actions.copy(); actions[:, 8] = 0.; action_mask = np.zeros((8,16), bool); action_mask[:,8] = True; forced = np.full(8,8)
            elif reason in ('HOLD', 'SETTLING'):
                for goal in range(8):
                    if forced[goal] < 0:
                        forced[goal] = 0 if action_mask[goal,0] else int(np.flatnonzero(action_mask[goal])[0])
            if reason=='WAIT_TASK':
                action_mask[:]=False;action_mask[:,8]=True;forced[:]=8
            features = []
            for endpoint, route in candidates:
                features.append(np.r_[(endpoint-p)/bounds, (c.task-p)/bounds, (route.tracking(p)-p)/bounds,
                    np.eye(4)[phase], min(age, 400)/400., route.index/max(1, len(route.points)), np.eye(4)[i]])
            # Rule recovery uses PLANNED distance and time, never only Euclidean progress.
            scores = [np.linalg.norm(point-c.task)+.1*r.estimated_seconds for point, r in candidates]
            valid = np.flatnonzero(goal_mask)
            if reason == 'STALLED' and any(valid != 0):
                valid = valid[valid != 0]
            rule = int(valid[np.argmin(np.array(scores)[valid])])
            rule_end = self.manager.rule_terminate(reason, age)
            period_end = self.manager.rule_terminate(reason, age, periodic=True)
            hard, allowed = self.manager.termination_flags(reason, age, phase, self._has_choices[i])
            if self.intervention != 'none':
                raise ValueError('R123 requires explicit retrained arms, not old-policy interventions')
            rows.append(dict(history=packet.features[i].reshape(self.o['history'], 90).copy(),
                paths=[r.tokens(state, p, bounds) for _, r in candidates], goals=np.asarray(features, np.float32),
                actions=actions, action_mask=action_mask, goal_mask=goal_mask, forced=forced,
                event_relative=np.asarray([(t-p)/bounds for t in tracks], np.float32),
                event_targets=np.array(tracks), position=p.copy(), velocity=packet.velocities[i].copy(), bounds=bounds.copy(),
                phase=phase, reason=reason, age=age, rule_goal=rule,
                rule_terminate=rule_end, periodic_terminate=period_end, forced_terminate=hard,
                beta_allowed=allowed, intervention=self.intervention, emergency=emergency,
                original=raw.copy(), task=c.task.copy(), goal_id=c.goal_id, route_id=c.route.route_id,
                route_index=c.route.index, route_revision=c.route.revision, map_revision=state.map_revision,
                cross_track=c.route.cross_track(p), estimated_seconds=c.route.estimated_seconds,
                endpoint=c.endpoint.copy(), free_pose=bool(packet.pose_free[i]), clearance=float(packet.clearances[i])))
            if self.switches['r2']:
                rows[-1]['rule_terminate'] = reason in ('REGION_ARRIVED','KNOWN_INVALID','NO_ROUTE','STALLED','RECOVERY_EXPIRED')
                rows[-1]['forced_terminate'] = rows[-1]['rule_terminate'] and self._has_choices[i]
                rows[-1]['beta_allowed'] = False
            elif reason=='RECOVERY_EXPIRED':
                rows[-1]['forced_terminate']=bool(self._has_choices[i]);rows[-1]['beta_allowed']=False
            if reason in ('WAIT_TASK','TASK_ONLY'):
                rows[-1].update(rule_terminate=False,forced_terminate=False,beta_allowed=False)
            rows[-1]['task_id'] = self.lifecycle.task_id(i)
            rows[-1]['target_velocity'] = packet.target_velocity.copy()
            rows[-1]['committed_original'] = c.original.copy()
            rows[-1]['action_diagnostics'] = bank_diagnostics
            assignment=self.assignment(i)
            rows[-1]['assignment_kind']=assignment['kind']
            rows[-1]['assignment_token']=assignment['token']
        for i,row in enumerate(rows):
            if row['phase']==3 or row['reason'] in ('WAIT_TASK','CONNECTOR','TRACK_ROOT'):
                row.update(rule_terminate=False,forced_terminate=False,beta_allowed=False)
            if row['reason']=='WAIT_TASK':
                result=refined_buffer_exit(packet.positions[i],packet.velocities[i],SPECS[i],geometry,self.guard.contract,self.guard.options,packet.flow[i])
                if result is not None:
                    row['actions'][:,9]=result[0]/limits(SPECS[i])
                    row['action_mask'][:]=False;row['action_mask'][:,9]=True;row['forced'][:]=9
        self._cached = (self.step, rows)
        return rows

    def _options(self,i,state,packet):
        result=super()._options(i,state,packet)
        c=result[0]
        c.route.tracking_owner=(int(packet.phases[i]),self.lifecycle.task_id(i))
        return result

    def _query(self,i,point,state):
        fresh=super()._query(i,point,state)
        if not self.tracking_switches['path']:return fresh
        c=self.manager.current.get(i)
        task=self.lifecycle.tasks.get(i)
        if c is None or task is None or task.released or task.phase==3:return fresh
        same_owner=getattr(c.route,'tracking_owner',None)==(task.phase,task.identity)
        same_target=np.linalg.norm(np.asarray(point)-c.endpoint)<=1e-4
        if not same_owner or (task.phase!=2 and not same_target):return fresh
        fresh.tracking_owner=(task.phase,task.identity)
        outcome=carry_progress(c.route,fresh,np.asarray(state.agents[i].position),self._geometry,self.o['arrival_radius'])
        self.route_events.append(dict(agent=i,step=state.step,task_id=task.identity,phase=task.phase,
                                      old_route=c.route.route_id,new_route=fresh.route_id,**outcome))
        return fresh

    def summary(self):
        result=super().summary();result['method']='auv.r123.tracking_rule.v1'
        result['path_option']['tracking_arm']=self.tracking_arm
        result['path_option']['tracking_switches']=self.tracking_switches.copy()
        result['path_option']['route_continuity_events']=self.route_events.copy()
        return result
