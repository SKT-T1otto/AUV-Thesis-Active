"""Finite joint map/target filter. Inputs are public evidence, never an env."""
from collections import Counter, deque
import numpy as np
from core.env.target_motion import TargetState, advance_target_state
from .geometry_hypotheses import GeometryHypothesisBank, compatible
from .geometry_cache import ObservationCache
from .numerics import Budget, BudgetExceeded, digest, locked, probabilities, swept_distances
from .types import TargetForecast
from .target_particles import propagate
from .diagnostics import StageAudit, observe_inference


class CoupledBelief:
    def __init__(self, config, contract, vehicle_specs):
        self.config,self.contract,self.specs=config,contract,vehicle_specs
        self.rng=np.random.default_rng(config['seed'])
        self.scan_cache=ObservationCache(config['history_steps']+1)
        self.geometry=GeometryHypothesisBank(config,contract,self.rng,scan_cache=self.scan_cache)
        self.history=deque(maxlen=config['history_steps']+1)
        self.history_complete=True
        self.seen=set();self.counts=Counter();self.step=0
        self.maps=();self.positions=None;self.velocities=None
        self.alpha=None;self.omega=None
        self.valid=False;self.error='not_initialized';self.stopped=False;self.retry_at=0
        self.first_failure=None;self.last_failure=None
        self.inference_diagnostics=StageAudit(config['audit_capacity'])

    def initialize_hypotheses(self, maps, positions, velocities, alpha, omega, *, step=0):
        """Explicit finite prior injection, also used by independent analytic checks."""
        p=np.asarray(positions,float);v=np.asarray(velocities,float)
        a=probabilities(alpha);w=np.asarray([probabilities(x) for x in omega])
        if (p.shape!=v.shape or p.ndim!=3 or p.shape[0]!=len(maps) or p.shape[2]!=3
                or w.shape!=p.shape[:2] or a.shape!=(len(maps),)
                or not np.all(np.isfinite(p)) or not np.all(np.isfinite(v))):
            raise ValueError('joint prior shape/finite contract')
        bounds=np.asarray(self.contract['bounds'])
        for m,x in zip(maps,p):
            if np.any(x<0) or np.any(x>bounds) or not np.all(m.free(x,self.contract['target_obstacle_clearance'])):
                raise ValueError('target prior outside hypothesized free space')
        self.maps=tuple(maps);self.positions=p.copy();self.velocities=v.copy()
        self.alpha=a;self.omega=w;self.step=int(step);self.valid=True;self.error=None

    def _prior(self, maps, budget):
        cfg=self.config;n=cfg['particles_per_map'];bounds=np.asarray(self.contract['bounds'])
        pp=[];vv=[];prior=cfg['target_prior']
        for geometry in maps:
            points=[]
            for _ in range(256):
                budget.check()
                pool=self.rng.uniform(0,bounds,(max(n,16),3))
                points.extend(pool[geometry.free(pool,self.contract['target_obstacle_clearance'])])
                if len(points)>=n:break
            if len(points)<n:raise ValueError('target_prior_free_space_exhausted')
            direction=self.rng.normal(size=(n,3));direction/=np.linalg.norm(direction,axis=1)[:,None]
            speed=self.rng.uniform(prior['speed_min'],prior['speed_max'],n)
            speed[self.rng.random(n)<prior['stationary_mass']]=0
            pp.append(points[:n]);vv.append(direction*speed[:,None])
        self.initialize_hypotheses(maps,pp,vv,np.ones(len(maps)),np.ones((len(maps),n)))

    def _propagate(self, p, v, maps, step, budget):
        new_p=np.empty_like(p);new_v=np.empty_like(v)
        for r,m in enumerate(maps):
            new_p[r],new_v[r]=propagate(p[r],v[r],m,step,self.contract,budget)
        return new_p,new_v

    def _update(self, evidence, budget, *, replay=False):
        if evidence.physical_step not in (self.step,self.step+1):raise ValueError('joint_filter_step_gap')
        lm=np.ones(len(self.maps))
        for event in evidence.scans:
            for r,m in enumerate(self.maps):
                budget.check()
                lm[r]*=float(self.scan_cache.compatible(m,event.payload,self.contract['bounds'],
                    self.contract['obstacle_sensor_range'],self.config['geometry']['scan_tolerance'],compatible))
        transition=evidence.transition
        p,v=self.positions,self.velocities
        ln=np.ones_like(self.omega)
        if transition is not None and transition.payload.target_transition_completed:
            p,v=self._propagate(self.positions,self.velocities,self.maps,self.step,budget)
            if evidence.negative_observation_valid:
                obs=transition.payload
                for i in range(3):
                    distance=swept_distances(np.asarray(obs.agent_start[i]),np.asarray(obs.agent_end[i]),self.positions,p)
                    ln*=distance>self.specs[i]['sensor_range']+self.contract['detect_eps_bias']
        raw=self.omega*ln
        z=raw.sum(axis=1)
        joint=self.alpha*lm*z
        if not np.isfinite(joint.sum()) or joint.sum()<=0:
            raise ValueError('joint_evidence_zero_mass')
        # No miss floor: zero conditional modes carry exactly zero map weight.
        omega=np.divide(raw,z[:,None],out=np.zeros_like(raw),where=z[:,None]>0)
        self.positions,self.velocities=p,v;self.alpha=joint/joint.sum();self.omega=omega
        self.step=evidence.physical_step
        if transition is not None:
            obs=transition.payload
            self.stopped=bool(obs.found_after or obs.episode_terminated)
        if not replay:self.counts['joint_updates']+=1

    def _rebuild(self, budget):
        if not self.history_complete:raise ValueError('replay_history_truncated')
        scans=[e.payload for item in self.history for e in item.scans]
        if not self.history or self.history[0].physical_step!=0:
            raise ValueError('replay_requires_initial_public_scan')
        self.inference_diagnostics.stage('geometry_rebuild')
        maps=self.geometry.propose(scans,budget)
        saved=(self.maps,self.positions,self.velocities,self.alpha,self.omega,self.step,self.valid,self.stopped)
        try:
            self.inference_diagnostics.stage('prior')
            self._prior(maps,budget)
            self.stopped=False
            self.inference_diagnostics.stage('full_history_replay')
            for item in self.history:self._update(item,budget,replay=True)
            self.counts['full_history_replays']+=1
        except (ValueError,RuntimeError):
            (self.maps,self.positions,self.velocities,self.alpha,self.omega,self.step,self.valid,self.stopped)=saved
            raise

    @observe_inference
    def consume(self, evidence):
        ids=tuple(e.event_id for e in evidence.scans)+(() if evidence.transition is None else (evidence.transition.event_id,))
        if ids and all(e in self.seen for e in ids):
            self.counts['duplicate_packets']+=1;return
        if any(e in self.seen for e in ids):raise ValueError('partial_duplicate_joint_packet')
        if self.stopped:return
        self.seen.update(ids)
        if len(self.history)==self.history.maxlen:self.history_complete=False
        self.history.append(evidence)
        budget=Budget(self.config['inference_seconds'])
        if not self.valid and evidence.physical_step<self.retry_at:return
        try:
            self.inference_diagnostics.stage('incremental_update')
            if self.valid:self._update(evidence,budget)
            else:self._rebuild(budget)
            self.valid=True;self.error=None
            return
        except (ValueError,RuntimeError) as error:
            self.inference_diagnostics.note_failure(error)
            self.valid=False;self.error=str(error);self.counts['inference_failures']+=1
            self.last_failure=dict(step=evidence.physical_step,reason=str(error),phase='incremental' if self.maps else 'initialization')
            if self.first_failure is None:self.first_failure=dict(self.last_failure)
        for _ in range(self.config['replenish_attempts']):
            try:
                budget.check();self._rebuild(budget)
                self.valid=True;self.error=None;self.counts['replenishments']+=1;return
            except (ValueError,RuntimeError) as error:
                self.inference_diagnostics.note_failure(error)
                self.error=str(error)
        self.retry_at=evidence.physical_step+self.config['recovery_cooldown']

    @property
    def identity(self):
        if not self.valid:return None
        return digest(dict(step=self.step,maps=[m.identity for m in self.maps],
            p=self.positions.tolist(),v=self.velocities.tolist(),a=self.alpha.tolist(),w=self.omega.tolist()))

    def forecast(self, budget):
        if not self.valid or self.stopped:raise ValueError('coupled_belief_unavailable:'+str(self.error))
        p,v=self.positions.copy(),self.velocities.copy();trajectory=[p.copy()]
        for k in range(self.config['horizon']):
            p,v=self._propagate(p,v,self.maps,self.step+k,budget);trajectory.append(p.copy())
        positions=np.stack(trajectory,axis=2).reshape(-1,self.config['horizon']+1,3)
        weights=(self.alpha[:,None]*self.omega).reshape(-1)
        active=weights>0
        maps=tuple(r for r in range(len(self.maps)) for _ in range(self.omega.shape[1]))
        identity=digest(dict(belief=self.identity,horizon=self.config['horizon']))
        return TargetForecast(locked(positions[active]),locked(probabilities(weights[active])),
                              tuple(r for r,ok in zip(maps,active) if ok),self.step,identity)

    def summary(self):
        return dict(valid=self.valid,error=self.error,step=self.step,stopped=self.stopped,
            alpha=None if self.alpha is None else self.alpha.tolist(),
            conditional_ess=None if self.omega is None else [float(1/np.sum(w*w)) if np.any(w) else 0. for w in self.omega],
            history_complete=self.history_complete,retained_history=len(self.history),counts=dict(self.counts),
            geometry_attempts=self.geometry.attempts,prior=self.config['geometry']['prior_id'],
            support_boxes=self.geometry.support_boxes,distinct_maps=self.geometry.distinct_maps,
            constraint_rays=self.geometry.constraint_rays,geometry_rebuilds=self.geometry.rebuilds,
            first_failure=self.first_failure,last_failure=self.last_failure)
