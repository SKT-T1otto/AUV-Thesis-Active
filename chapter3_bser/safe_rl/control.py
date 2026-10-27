"""All-agent physical-step speed governor, recovery arbitration and trace."""
from collections import Counter,deque
import numpy as np
from chapter3_bser.safe_rl.options import FRAME_DIM
from chapter3_bser.safe_rl.dynamics import prior,nominal,limits,brake,integrate,stopping_bound,speed_limit
from chapter3_bser.safe_rl.evidence import ScanEvidence,EvidenceGeometry


class SafetyController:
    def __init__(self,specs,contract,options,*,learner=None):
        self.specs=specs;self.contract=contract;self.options=options;self.learner=learner
        self.evidence=ScanEvidence();self.counts=Counter();self.trace=[]
        self.histories=[deque(maxlen=options['history']) for _ in range(4)]
        self.previous=np.zeros((4,3));self.previous_v=None;self.flow=np.zeros((4,2))
        self.recovering=np.zeros(4,bool);self.release=np.zeros(4,int);self.blocked=np.zeros(4,int)
        self.anchors=[None]*4

    def observe(self,state,events):
        self.evidence.consume(events)
        velocity=np.asarray([a.velocity for a in state.agents])
        if self.previous_v is not None:
            dt=self.contract['dt'];bound=self.options['flow_component_bound']
            for i,spec in enumerate(self.specs):
                # Skip clipped velocity: it cannot identify disturbance correctly.
                if np.linalg.norm(velocity[i,:2])>=spec['v_xy_max']-1e-5:continue
                inferred=(velocity[i,:2]-(1-spec['drag_xy']*dt)*(self.previous_v[i,:2]+self.previous[i,:2]*dt))/dt
                self.flow[i]=np.clip(inferred,-bound,bound)
        return EvidenceGeometry(state,self.evidence,self.options)

    def _caps(self,p,v,spec,geometry):
        o=self.options;dt=self.contract['dt'];sensor=self.contract['obstacle_sensor_range']
        gap=min(sensor,geometry.distance(p));unknown,free=geometry.local(p)
        available=max(0.,gap-o['clearance']-o['hit_padding'])
        if unknown>0 or not free:available*=o['uncertainty_slowdown']
        return (speed_limit(available,spec,dt,o['flow_component_bound'],reaction_steps=o['reaction_steps']),
                speed_limit(available,spec,dt,o['flow_component_bound'],vertical=True,reaction_steps=o['reaction_steps']),available)

    def _features(self,i,state,item,semantic,command,geometry,caps):
        s=self.specs[i];o=self.options;dt=self.contract['dt'];radius=self.contract['obstacle_sensor_range']
        a=state.agents[i];p=np.asarray(a.position);v=np.asarray(a.velocity);hit,voxel=geometry.point_distances(p)
        unknown,free=geometry.local(p);bounds=np.asarray(self.contract['bounds'])
        phase=[float(not state.target_found),float(state.target_found and not item.execution_request),float(item.execution_request)]
        parameters=[s['a_xy_max']/1.3,s['a_z_max']/.75,s['v_xy_max']/2.8,s['v_z_max']/1.2,
                    s['drag_xy'],s['drag_z'],s['buoyancy_bias']]
        frame=np.concatenate((p/bounds,v/np.array([s['v_xy_max']]*2+[s['v_z_max']]),
            (np.asarray(item.tracking_waypoint)-p)/radius,(np.asarray(semantic)-p)/radius,
            command/limits(s),self.previous[i]/limits(s),np.eye(4)[i],phase,parameters,
            [min(hit,radius)/radius,min(voxel,radius)/radius,unknown,float(free),
             min(1.,max(0,state.step-self.evidence.last_scan_step)/20),
             min(1.,getattr(self,'cache_age',0)/20),min(1.,len(self.evidence.hits)/1024),
             min(1.,state.map_revision/10000)],
            [caps[0]/s['v_xy_max'],caps[1]/s['v_z_max'],
             min(2.,stopping_bound(np.linalg.norm(v[:2]),s,dt,o['flow_component_bound'],reaction_steps=o['reaction_steps'])/radius),
             min(2.,stopping_bound(abs(v[2]),s,dt,o['flow_component_bound'],vertical=True,reaction_steps=o['reaction_steps'])/radius)],
            [float(item.hold_state),float(item.reachable),float(geometry.distance(semantic)<=geometry.padding),
             float(self.evidence.last_scan_step==state.step)]))
        if frame.shape!=(FRAME_DIM,) or not np.isfinite(frame).all():raise ValueError('invalid safety feature contract')
        history=self.histories[i];history.append(frame.astype(np.float32))
        while len(history)<o['history']:history.appendleft(frame.astype(np.float32))
        return np.concatenate(tuple(history)).astype(np.float32)

    def check(self,p,v,command,spec,geometry,flow):
        """Finite public-model motion+backup check, never a truth certificate."""
        dt=self.contract['dt'];o=self.options;p=np.asarray(p,float);v=np.asarray(v,float)
        old=p.copy();travel=0.;stopped=False
        for k in range(o['horizon']):
            u=command if k==0 else brake(v,spec,dt,flow)
            nxt,vel=integrate(p,v,u,spec,dt,flow)
            # Conservative growing disturbance envelope for this finite probe.
            # Feedback/partial sensing prevent interpreting it as a formal invariant.
            extra=np.sqrt(2)*2*o['flow_component_bound']*dt*dt*(k+1)*(k+2)/2
            if not geometry.segment(p,nxt,extra=extra):return False,dict(reason='motion_or_backup_geometry',probe_steps=k+1)
            travel+=float(np.linalg.norm(nxt-p));p,v=nxt,vel
            if k>=1 and np.linalg.norm(v)<.04:stopped=True;break
        return stopped,dict(reason='model_checked_stop' if stopped else 'backup_horizon_exhausted',
            probe_steps=k+1,predicted_travel=travel,predicted_stop=p.tolist(),
            terminal_invariant_certified=False,ground_truth_geometry_used=False)

    def act(self,state,guidance,task_actions,events,*,semantic_goals=None,cache_age=0):
        self.cache_age=cache_age;geometry=self.observe(state,events);outputs=[];rows=[];features=[]
        assignments={x.agent_id:x for x in guidance.agent_assignments};dt=self.contract['dt'];o=self.options
        semantic_goals=semantic_goals or {i:x.final_waypoint for i,x in assignments.items()}
        for i,spec in enumerate(self.specs):
            a=state.agents[i];item=assignments[i];p=np.asarray(a.position);v=np.asarray(a.velocity)
            target=item.hold_position if item.hold_state else item.tracking_waypoint
            u0,base,residual=nominal(p,v,target,task_actions[i],spec,self.contract,i)
            caps=self._caps(p,v,spec,geometry)
            z=self._features(i,state,item,semantic_goals[i],u0,geometry,caps);features.append(z)
            ugovern=prior(p,v,target,spec,self.contract,speed_xy=caps[0],speed_z=caps[1])+residual
            ugovern=np.clip(ugovern,-limits(spec),limits(spec))
            predicted=integrate(p,v,u0,spec,dt,self.flow[i])[1]
            speed_ok=np.linalg.norm(predicted[:2])<=caps[0]+1e-6 and abs(predicted[2])<=caps[1]+1e-6
            original_ok,probe=self.check(p,v,u0,spec,geometry,self.flow[i])
            risk=None;learned=None
            if self.learner is not None and self.learner.ready:
                risk=self.learner.risk(z,u0/limits(spec))
                learned=self.learner.action(z)*limits(spec)
            goal_conflict=geometry.distance(item.final_waypoint)<=geometry.padding
            safe=original_ok and speed_ok and not goal_conflict and (risk is None or risk['score']<o['risk_threshold'])
            self.release[i]=self.release[i]+1 if safe else 0
            if not safe:self.recovering[i]=True
            if self.release[i]>=o['release_steps']:self.recovering[i]=False
            accepted=bool(safe and not self.recovering[i]);chosen=u0;mode='NOMINAL';check=probe
            if not accepted:
                mode='RECOVERY';self.blocked[i]+=1
                anchor=self.anchors[i]
                anchor_acc=brake(v,spec,dt,self.flow[i]) if anchor is None else prior(p,v,anchor,spec,self.contract,speed_xy=caps[0],speed_z=caps[1])
                outward=geometry.outward(p);norm=float(np.linalg.norm(outward))
                outward=outward/max(norm,1e-9)
                escape=np.clip(brake(v,spec,dt,self.flow[i])+outward*limits(spec),-limits(spec),limits(spec))
                candidates=[]
                if learned is not None:candidates.append(('LEARNED_RECOVERY',learned))
                candidates.extend([('GOVERNED',ugovern),('ANCHOR',anchor_acc),('BRAKE',brake(v,spec,dt,self.flow[i])),('ESCAPE',escape)])
                violation=max(0.,np.linalg.norm(v[:2])-caps[0])+max(0.,abs(v[2])-caps[1])
                for label,u in candidates:
                    next_velocity=integrate(p,v,u,spec,dt,self.flow[i])[1]
                    next_violation=max(0.,np.linalg.norm(next_velocity[:2])-caps[0])+max(0.,abs(next_velocity[2])-caps[1])
                    if next_violation>1e-6 and next_violation>=violation-1e-8:continue
                    ok,detail=self.check(p,v,u,spec,geometry,self.flow[i])
                    if ok and self.learner is not None and self.learner.ready:
                        candidate_risk=self.learner.risk(z,np.asarray(u)/limits(spec))
                        detail['learned_candidate_risk']=candidate_risk
                        if candidate_risk['score']>=o['risk_threshold']:continue
                    if ok:chosen=u;mode=label;check=detail;accepted=True;break
                if not accepted:
                    # Never label emergency risk minimization as checked/safe.
                    chosen=brake(v,spec,dt,self.flow[i]);mode='UNCERTIFIED_BRAKE'
                    check=dict(reason='no_model_checked_candidate',terminal_invariant_certified=False)
            else:self.blocked[i]=0
            if accepted and geometry.distance(p)>geometry.padding+.5 and np.linalg.norm(v)<.25:
                self.anchors[i]=p.copy()
            self.counts['agent_checks']+=1;self.counts[f'agent_{i}:{mode}']+=1
            row=dict(step=int(state.step),agent_id=i,mode=mode,model_checked=accepted,
                observation_space_certified=False,safety_guarantee=False,physical_position=p.tolist(),velocity=v.tolist(),
                semantic_goal=list(semantic_goals[i]),navigation_goal=list(item.final_waypoint),observed_goal_conflict=bool(goal_conflict),tracking=list(target),nominal_acceleration=u0.tolist(),
                prior=base.tolist(),task_residual=residual.tolist(),governed_acceleration=ugovern.tolist(),
                executed_acceleration=np.asarray(chosen).tolist(),speed_cap_xy=caps[0],speed_cap_z=caps[1],
                available_evidence_distance=caps[2],risk=risk,check=check,
                current_speed_exceeds_cap=bool(np.linalg.norm(v[:2])>caps[0] or abs(v[2])>caps[1]),
                map_revision=int(state.map_revision),scan_age=max(0,int(state.step)-self.evidence.last_scan_step))
            rows.append(row);outputs.append(np.asarray(chosen)/limits(spec))
        self.previous=np.array([r['executed_acceleration'] for r in rows]);self.previous_v=np.array([a.velocity for a in state.agents])
        self.trace.extend(rows)
        return np.asarray(outputs,np.float32),np.asarray(features,np.float32),rows,geometry
