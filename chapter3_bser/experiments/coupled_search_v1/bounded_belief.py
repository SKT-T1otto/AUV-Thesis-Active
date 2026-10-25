"""V3 persistent map / bounded raw window / boundary posterior / local repair.

Transport across changed map geometry and jitter proposals are declared finite
posterior approximations. There is no uniform reset in the full algorithm and
no miss floor. A recovery that cannot satisfy the retained evidence fails.
"""
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass
import time
import numpy as np
from .coupled_belief import CoupledBelief
from .persistent_map import PersistentVoxelMap,LocalGeometryBank
from .numerics import Budget,digest
from .target_particles import propagate
from .diagnostics import observe_inference


@dataclass(frozen=True)
class BoundaryJointState:
    step:int
    maps:tuple
    positions:np.ndarray
    velocities:np.ndarray
    alpha:np.ndarray
    omega:np.ndarray

    @property
    def nbytes(self):
        return sum(a.nbytes for a in (self.positions,self.velocities,self.alpha,self.omega))


class BoundedCoupledBelief(CoupledBelief):
    def __init__(self,config,contract,vehicle_specs):
        super().__init__(config,contract,vehicle_specs)
        self.memory=PersistentVoxelMap(config,contract)
        self.geometry=LocalGeometryBank(config,contract,self.rng,self.memory)
        self.history=deque();self.snapshots=deque();self.boundary=None
        self.map_frames=deque(maxlen=config['history_steps'])
        self.seen=OrderedDict();self.timings=deque(maxlen=config['audit_capacity'])
        self.window=config['history_steps'];self.last_replay_steps=0;self.max_replay_steps=0
        self.last_replay_work=0
        self.max_retained_history=0;self.max_boxes=0;self.counts=Counter()

    def _snapshot(self):
        arrays=[]
        for value in (self.positions,self.velocities,self.alpha,self.omega):
            copied=value.copy();copied.setflags(write=False);arrays.append(copied)
        return BoundaryJointState(self.step,self.maps,*arrays)

    def _restore(self,state,*,maps=None):
        self.step=state.step;self.maps=state.maps if maps is None else maps
        self.positions=state.positions.copy();self.velocities=state.velocities.copy()
        self.alpha=state.alpha.copy();self.omega=state.omega.copy()

    def _propagate(self,p,v,maps,step,budget):
        pp=p.copy();vv=v.copy()
        for index,geometry in enumerate(maps):
            active=(self.omega[index]>0)&(self.alpha[index]>0)
            if np.any(active):
                pp[index,active],vv[index,active]=propagate(p[index,active],v[index,active],geometry,step,self.contract,budget)
        return pp,vv

    def _transport(self,maps):
        mass=np.zeros(len(maps));weights=np.zeros_like(self.omega)
        for index,geometry in enumerate(maps):
            free=geometry.free(self.positions[index],self.contract['target_obstacle_clearance'])
            raw=self.omega[index]*free;mass[index]=raw.sum()
            if mass[index]>0:weights[index]=raw/mass[index]
        alpha=self.alpha*mass
        if alpha.sum()<=0:raise ValueError('bounded_geometry_transport_zero_mass')
        self.maps=maps;self.alpha=alpha/alpha.sum();self.omega=weights
        self.counts['approximate_geometry_transports']+=1

    def _resample(self):
        threshold=self.config['bounded']['resample_ess_fraction']*self.config['particles_per_map']
        for index,weights in enumerate(self.omega):
            if self.alpha[index]<=0 or not np.any(weights):continue
            ess=1/np.sum(weights*weights)
            if ess>=threshold:continue
            n=len(weights);cdf=np.cumsum(weights);cdf[-1]=1.
            selected=np.searchsorted(cdf,(self.rng.random()+np.arange(n))/n,side='right')
            self.positions[index]=self.positions[index,selected];self.velocities[index]=self.velocities[index,selected]
            self.omega[index]=1./n;self.counts['conditional_resamplings']+=1

    def _target_update(self,evidence,budget):
        if evidence.physical_step!=self.step+1:raise ValueError('bounded_filter_step_gap')
        p,v=self.positions,self.velocities;likelihood=np.ones_like(self.omega)
        transition=evidence.transition
        if transition is not None and transition.payload.target_transition_completed:
            from .numerics import swept_distances
            p,v=self._propagate(p,v,self.maps,self.step,budget)
            if evidence.negative_observation_valid:
                obs=transition.payload
                for i in range(3):
                    distance=swept_distances(np.asarray(obs.agent_start[i]),np.asarray(obs.agent_end[i]),self.positions,p)
                    likelihood*=distance>self.specs[i]['sensor_range']+self.contract['detect_eps_bias']
        raw=self.omega*likelihood;mass=raw.sum(axis=1);joint=self.alpha*mass
        if not np.isfinite(joint.sum()) or joint.sum()<=0:raise ValueError('bounded_target_zero_mass')
        self.positions,self.velocities=p,v;self.alpha=joint/joint.sum()
        self.omega=np.divide(raw,mass[:,None],out=np.zeros_like(raw),where=mass[:,None]>0)
        self.step=evidence.physical_step;self._resample()
        if transition is not None:self.stopped=bool(transition.payload.found_after or transition.payload.episode_terminated)

    def _seed_from_boundary(self,maps,budget):
        boundary=self.boundary;b=self.config['bounded'];n=self.config['particles_per_map']
        if b['boundary_mode']=='uniform_ablation':
            self._prior(maps,budget);self.step=boundary.step
            self.counts['uniform_boundary_ablation_resets']+=1
            return
        self._restore(boundary,maps=maps)
        # Conditional proposals start exclusively from positive boundary mass.
        # Jitter is explicitly approximate, never a floor on rejected evidence.
        for index,weights in enumerate(boundary.omega):
            if boundary.alpha[index]<=0 or weights.sum()<=0:
                self.alpha[index]=0;self.omega[index]=0;continue
            budget.check()
            selected=self.rng.choice(n,n,p=weights/weights.sum())
            p=boundary.positions[index,selected]+self.rng.normal(0,b['position_jitter'],(n,3))
            v=boundary.velocities[index,selected]+self.rng.normal(0,b['velocity_jitter'],(n,3))
            p=np.clip(p,0,self.contract['bounds'])
            speed=np.linalg.norm(v,axis=1);maximum=self.config['target_prior']['speed_max']
            v*=np.minimum(1.,maximum/np.maximum(speed,1e-12))[:,None]
            free=maps[index].free(p,self.contract['target_obstacle_clearance'])
            mass=float(free.mean());self.alpha[index]*=mass
            self.positions[index]=p;self.velocities[index]=v
            self.omega[index]=free/free.sum() if np.any(free) else 0.
        if self.alpha.sum()<=0:raise ValueError('boundary_proposal_free_space_exhausted')
        self.alpha/=self.alpha.sum();self.counts['boundary_conditioned_proposals']+=1

    def _recover(self,maps,budget):
        if not self.config['bounded']['local_recovery']:raise ValueError('local_recovery_disabled_ablation')
        if self.boundary is None:raise ValueError('boundary_joint_state_missing')
        error=None
        for _ in range(self.config['replenish_attempts']):
            budget.check();self.counts['local_recovery_attempts']+=1
            try:
                self.inference_diagnostics.stage('recovery_seed')
                self._seed_from_boundary(maps,budget);self.stopped=False;snapshots=[];attempt_steps=0
                self.inference_diagnostics.stage('window_replay')
                for item in self.history:
                    budget.check();attempt_steps+=1;self.last_replay_work+=1
                    self.last_replay_steps=attempt_steps;self.max_replay_steps=max(self.max_replay_steps,attempt_steps)
                    self.counts['bounded_replay_steps']+=1
                    self._target_update(item,budget);snapshots.append(self._snapshot())
                self.snapshots=deque(snapshots)
                self.counts['local_recoveries']+=1
                return
            except (ValueError,RuntimeError) as failure:
                self.inference_diagnostics.note_failure(failure)
                error=failure
        raise ValueError('bounded_recovery_exhausted:'+str(error))

    @observe_inference
    def consume(self,evidence):
        ids=tuple(event.event_id for event in evidence.scans)+(() if evidence.transition is None else (evidence.transition.event_id,))
        if ids and all(event in self.seen for event in ids):self.counts['duplicate_packets']+=1;return
        if any(event in self.seen for event in ids):raise ValueError('partial_duplicate_joint_packet')
        if self.stopped:return
        if len(evidence.scans)>1:raise ValueError('bounded_scan_batch_capacity')
        if self.boundary is None and (evidence.physical_step!=0 or not evidence.scans):
            raise ValueError('initial_public_scan_required')
        if self.boundary is not None and evidence.physical_step!=self.step+1:raise ValueError('bounded_filter_step_gap')
        for event in ids:self.seen[event]=True
        while len(self.seen)>2*(self.window+1)+8:self.seen.popitem(last=False)
        start=time.perf_counter();budget=Budget(self.config['inference_seconds']);phase={}
        saved=self._snapshot() if self.valid else None
        if self.boundary is not None:
            if len(self.history)==self.window:
                self.history.popleft();self.boundary=self.snapshots.popleft();self.counts['boundary_advances']+=1
            self.history.append(evidence)
        self.max_retained_history=max(self.max_retained_history,len(self.history));self.last_replay_steps=0;self.last_replay_work=0
        try:
            phase_start=time.perf_counter()
            self.inference_diagnostics.stage('scan_and_map_memory')
            if self.config['bounded']['map_memory']=='window_ablation':
                frame=self.memory.observe([event.payload for event in evidence.scans],budget)
                self.map_frames.append(frame)
                self.memory.replace_window(self.map_frames,frame,budget)
            else:self.memory.ingest([event.payload for event in evidence.scans],budget)
            self.inference_diagnostics.stage('geometry_repair')
            maps,changed=self.geometry.repair(budget);phase['map_seconds']=time.perf_counter()-phase_start
            self.max_boxes=max(self.max_boxes,max(map(len,(geometry.boxes for geometry in maps)),default=0))
            budget.check();phase_start=time.perf_counter()
            self.inference_diagnostics.stage('target_update')
            if self.boundary is None:
                if evidence.physical_step!=0:raise ValueError('initial_public_scan_required')
                self._prior(maps,budget);self.boundary=self._snapshot();self.counts['initializations']+=1
            else:
                try:
                    if changed:self._transport(maps)
                    else:self.maps=maps
                    self._target_update(evidence,budget)
                    self.snapshots.append(self._snapshot())
                except ValueError as failure:
                    self.inference_diagnostics.note_failure(failure)
                    if str(failure) not in ('bounded_target_zero_mass','bounded_geometry_transport_zero_mass'):raise
                    self.counts['local_recovery_triggers']+=1
                    if self.first_failure is None:self.first_failure=dict(step=evidence.physical_step,reason=str(failure),phase='local')
                    self._recover(maps,budget)
            phase['target_and_recovery_seconds']=time.perf_counter()-phase_start
            budget.check();self.valid=True;self.error=None;self.counts['joint_updates']+=1
        except (ValueError,RuntimeError) as error:
            self.inference_diagnostics.note_failure(error)
            if saved is not None:self._restore(saved)
            self.valid=False;self.error=str(error);self.counts['inference_failures']+=1
            self.last_failure=dict(step=evidence.physical_step,reason=str(error),phase='bounded')
            if self.first_failure is None:self.first_failure=dict(self.last_failure)
        finally:
            self.timings.append(dict(step=evidence.physical_step,seconds=time.perf_counter()-start,
                retained_history=len(self.history),boundary_step=None if self.boundary is None else self.boundary.step,
                replay_steps=self.last_replay_steps,replay_work=self.last_replay_work,map_boxes=self.max_boxes,valid=self.valid,**phase))

    def summary(self):
        return dict(valid=self.valid,error=self.error,step=self.step,stopped=self.stopped,
            alpha=None if self.alpha is None else self.alpha.tolist(),history_complete=False,
            conditional_ess=None if self.omega is None else [float(1/np.sum(w*w)) if np.any(w) else 0. for w in self.omega],
            historical_information_carried_by_boundary=self.config['bounded']['boundary_mode']=='posterior',retained_history=len(self.history),window_steps=self.window,
            boundary_step=None if self.boundary is None else self.boundary.step,max_retained_history=self.max_retained_history,
            last_replay_steps=self.last_replay_steps,max_replay_steps=self.max_replay_steps,max_map_boxes=self.max_boxes,
            last_replay_work=self.last_replay_work,
            boundary_array_bytes=0 if self.boundary is None else self.boundary.nbytes,
            snapshot_array_bytes=sum(state.nbytes for state in self.snapshots),
            retained_snapshots=len(self.snapshots),retained_event_ids=len(self.seen),counts=dict(self.counts),
            window_map_frame_bytes=sum(frame.nbytes for frame in self.map_frames),
            persistent_map=self.memory.summary(),geometry_counts=dict(self.geometry.counts),
            first_failure=self.first_failure,last_failure=self.last_failure,
            approximate_map_transport=True,approximate_boundary_proposals=True,
            geometry_prior=self.config['geometry']['prior_id'],design_variant=self.config['bounded']['design_variant'])
