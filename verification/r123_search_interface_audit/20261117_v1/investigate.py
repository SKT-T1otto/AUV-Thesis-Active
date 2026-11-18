"""Read-only source/trace probes. Synthetic object checks, zero environments."""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
import numpy as np
torch.set_num_threads(1)

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def snap():
    roots=[ROOT.parent/n for n in ('CH3','CH4','CH5')]
    roots += [ROOT/n for n in ('core','chapter3_bser','tools','checks','configs','provenance','3090结果','verification/r123_search_failure_audit','verification/r123_tracking_result_audit')]
    return {p.relative_to(ROOT.parent).as_posix():sha(p) for base in roots for p in base.rglob('*') if p.is_file() and (base.name in ('CH3','CH4','CH5') or '__pycache__' not in p.parts)}

before=read(OUT/'before.json') if (OUT/'before.json').exists() else snap()
assert before==snap()
if not (OUT/'before.json').exists():write('before.json',before)

from checks.test_r123_repair import fixture,tick
from checks.test_r123 import observe
from chapter3_bser.d2.allocator import EffectAllocator
from chapter3_bser.online.allocator import BSEROnlineAllocator
from chapter3_bser.online.types import SearchAssignment,ExecutorAssignment,OnlineAllocation
from chapter3_bser.detection_model import detection_probability
from chapter3_bser.reach_avoid.options import SPECS,CONTRACT
from chapter3_bser.safe_rl.options import DEFAULTS as SAFETY
from chapter3_bser.safe_rl.dynamics import speed_limit
from chapter3_bser.config import load_bser_phase1a1_config
from chapter3_bser.candidate_generator import generate_search_candidates

probes={}
r,s,t=fixture('no_recovery');observe(r,s)
old=r.manager.current[0].task.copy();oldid=r.lifecycle.task_id(0)
item=r.controller.inner.current_allocation.search_assignments[0]
item.waypoint=(15.,3.,4.);item.candidate_id='synthetic_new'
row=tick(r,s,t,1)[0]
assert np.array_equal(row['task'],old) and np.array_equal(row['original'],item.waypoint)
assert r.lifecycle.task_id(0)==oldid
probes['active_task_keeps_old_root']=dict(passed=True,original=row['original'].tolist(),root=row['task'].tolist(),task_identity_unchanged=True)

allocation=OnlineAllocation(tuple(SearchAssignment(i,f's{i}',(9.,3.,4.),((1.,3.,4.),(9.,3.,4.)),8.,(0,1),8.) for i in range(3)),
    ExecutorAssignment(3,(7.,7.,4.),((7.,7.,4.),),0.,'synthetic',True,(),0.),1.,1.,1.,'synthetic')
owner=NS();allocator=EffectAllocator(owner,load_bser_phase1a1_config(),'D2',(7.,7.,4.))
captured=[]
def context(state,search,standby,config):
    captured.extend(search);return NS()
with patch('chapter3_bser.d2.allocator.build_objective_context',side_effect=context),patch('chapter3_bser.d2.allocator.evaluate_objective',return_value=1.),patch('chapter3_bser.d2.allocator.response_diagnostics',return_value=NS(conditional_reachable_response_time=1.)):
    allocator.rescore(NS(target_found=False),allocation)
assert captured[0].waypoint==(9.,3.,4.)
probes['rescore_uses_allocator_assignment']=dict(passed=True,scored_waypoint=list(captured[0].waypoint),
    scored_path=captured[0].path_points.tolist(),note='Synthetic executed root is 15,3,4; no lifecycle argument enters rescore')

newcandidate=BSEROnlineAllocator._frozen_search_candidate(allocation.search_assignments[2])
allocator.search_pool=lambda state,ids:((newcandidate,),0,())
allocator.standby_pool=lambda *args,**kwargs:((NS(),),allocation.executor_assignment)
seen={}
def proposal(state,candidates,frozen,*args):
    seen['frozen']=[dict(agent=c.agent_id,waypoint=list(c.waypoint),path=c.path_points.tolist()) for c in frozen]
    return allocation
allocator._proposal=proposal
_,ok,_=allocator.allocate_partial(NS(target_found=False,searcher_ids=(0,1,2)),allocation,affected_searcher_ids=(2,),executor_affected=False,trigger_reason='synthetic')
assert ok and [v['agent'] for v in seen['frozen']]==[0,1]
probes['partial_solve_freezes_allocator_assignments']=dict(passed=True,**seen)

# The actual spatial detection model retains a traversed prefix in a frozen path.
full=SearchAssignment(0,'full',(10.,0.,1.),((0.,0.,1.),(10.,0.,1.)),10.,(0,1),10.)
remaining=SearchAssignment(0,'remaining',(10.,0.,1.),((8.,0.,1.),(10.,0.,1.)),2.,(0,1),2.)
state=NS(agents=(NS(role='search_fast',sensor_radius=2.),),grid=NS(cell_centers=np.array([[0.,0.,1.],[10.,0.,1.]])),occupancy=NS(occupancy_probability=np.zeros(2)))
cfg=load_bser_phase1a1_config()
qfull=detection_probability(BSEROnlineAllocator._frozen_search_candidate(full),state,cfg)
qremaining=detection_probability(BSEROnlineAllocator._frozen_search_candidate(remaining),state,cfg)
assert qfull[0]>qremaining[0]
probes['frozen_path_keeps_past_prefix']=dict(passed=True,assumed_current_position=[8.,0.,1.],full_kernel=qfull.tolist(),remaining_kernel=qremaining.tolist(),scope='Synthetic scoring discrepancy; no success effect claimed')

shape=(10,10,8)
centers=np.array([[1.+2*x,1.+2*y,.5+z] for x in range(10) for y in range(10) for z in range(8)])
uniform=NS(grid=NS(shape=shape,cell_centers=centers),target_belief=NS(probabilities=np.full(800,1/800)),
    occupancy=NS(occupied_mask=np.zeros(800,bool),free_mask=np.zeros(800,bool),unknown_mask=np.ones(800,bool)),
    searcher_ids=(0,1,2),agents=tuple(NS(position=(18.,18.,4.)) for _ in range(3)))
def query(p,g,agent):
    length=float(np.linalg.norm(np.array(p)-g))
    return NS(reachable=True,physical_travel_time=length,path_points=np.array([p,g]),path_cell_indices=np.array([0]),path_length=length,planning_cost=length)
candidates,_,_=generate_search_candidates(uniform,NS(query=query),k_search=4,minimum_separation=.45,maximum_travel_time=120.)
assert len(candidates)==12 and all(c.waypoint[:2]==(1.,1.) for c in candidates)
probes['uniform_belief_candidate_tie_bias']=dict(passed=True,endpoints_by_agent={str(a):[list(c.waypoint) for c in candidates if c.agent_id==a] for a in range(3)},
    scope='Synthetic uniform all-unknown state with all queries reachable. Real initial belief/candidate pools were not logged, so real-scene cause remains unproven')
write('synthetic_probes.json',probes)

RUN=ROOT/'3090结果/r123_tracking_20261117_121047_2475513'
episodes=[json.loads(v) for v in (RUN/'both/episodes.jsonl').read_text(encoding='utf-8').splitlines()]
results=[];changes=[];slow=[];contexts=[];cap_errors=[]
for si,e in enumerate(episodes):
    tr=e['path_option']['trace'];byagent={a:[v for v in tr if v['agent']==a] for a in range(4)}
    ev=e['path_option']['lifecycle_events']
    info=dict(scene=si,changes=[],partial_release_contexts=[],continuity=Counter(v['reason'] for v in e['path_option']['route_continuity_events']))
    for a in range(3):
        for previous,current in zip(byagent[a],byagent[a][1:]):
            if current['phase']!=0:continue
            if previous['original']==current['original']:continue
            v=dict(scene=si,agent=a,step=current['step'],old_assignment=previous['assignment_token'],new_assignment=current['assignment_token'],both_assigned=previous['assignment_kind']==current['assignment_kind']=='search',root_before=previous['task'],root_now=current['task'],reason=current['reason'],position=current['position'],root_unchanged=previous['task']==current['task'],assignment_differs_root=current['original']!=current['task'])
            info['changes'].append(v);changes.append(v)
    for event in ev:
        if event['event']!='D2_RELEASE_REQUEST' or not event['accepted']:continue
        t0=event['step'];a=event['agent'];others=[]
        for b in range(3):
            if b==a:continue
            row=byagent[b][t0]
            if row['phase']==0 and row['assignment_kind']=='search' and row['original']!=row['task']:
                others.append(dict(agent=b,assignment=row['original'],executed_root=row['task'],reason=row['reason']))
        v=dict(scene=si,step=t0,requested_agent=a,other_mismatches=others,
               scope='Same-step trace context, combined with source freeze semantics; exact scoring graph/path was not logged')
        info['partial_release_contexts'].append(v);contexts.append(v)
    for row in tr:
        a=row['agent']
        if a>=3 or row['phase']!=0:continue
        p=np.array(row['position']);v=np.array(row['velocity']);goal=np.array(row['event_target'])
        if np.linalg.norm(v)>=.05 or np.linalg.norm(p-row['endpoint'])<=.8:continue
        d=row['action_diagnostics'];cap=d['speed_cap_xy']
        modelcap=speed_limit(d['available_distance'],SPECS[a],CONTRACT['dt'],SAFETY['flow_component_bound'],reaction_steps=SAFETY['reaction_steps'])
        assert abs(modelcap-cap)<1e-12
        cap_errors.append(abs(modelcap-cap))
        # This is the desired horizontal speed INSIDE prior, not actual velocity.
        desired=cap*min(float(np.linalg.norm(goal[:2]-p[:2]))/CONTRACT['prior_slow_radius_xy'],1.)
        point=dict(scene=si,agent=a,step=row['step'],reason=row['reason'],position=row['position'],tracking=row['event_target'],route_id=row['route_id'],
            cap_xy=cap,available_distance=d['available_distance'],distance_xy=float(np.linalg.norm(goal[:2]-p[:2])),
            desired_xy_inside_prior=desired,actual_xy=float(np.linalg.norm(v[:2])),action=row['decision']['action'])
        slow.append(point)
    results.append(info)
write('raw_followup.json',dict(cases=results,accepted_partial_events=len(contexts),accepted_partial_events_with_other_mismatch=sum(bool(v['other_mismatches']) for v in contexts),
    changes=len(changes),assignment_changes_keeping_root=sum(v['root_unchanged'] for v in changes),
    assigned_endpoint_changes=sum(v['both_assigned'] for v in changes),assigned_endpoint_changes_keeping_root=sum(v['both_assigned'] and v['root_unchanged'] for v in changes),
    slow_count=len(slow),slow_with_prior_desired_xy_below_05=sum(v['desired_xy_inside_prior']<.05 for v in slow),
    cap_reconstruction_max_error=max(cap_errors),synthetic_environments=0,physical_steps=0,training_updates=0,python=sys.executable,torch=torch.__version__,numpy=np.__version__))

base=[json.loads(v) for v in (ROOT/'3090结果/r123_governed_20261116_215327_3078331/governed/episodes.jsonl').read_text(encoding='utf-8').splitlines()]
capture=[]
for i in (3,4,5):
    case=dict(scene=i,arms={})
    for arm,e in [('baseline',base[i]),('both',episodes[i])]:
        rs=[r for r in e['path_option']['trace'] if r['agent']==3 and r['phase']==2]
        fixed=[r for r in rs if np.linalg.norm(np.array(r['event_target'])-r['endpoint'])>1e-4]
        case['arms'][arm]=dict(found_step=e['found_step'],received_step=e['executor_target_received_step'],success_step=e['success_step'],capture_steps=len(rs),
            fixed_waypoint_steps=len(fixed),other_tracking_steps=len(rs)-len(fixed),fixed_first_step=fixed[0]['step'] if fixed else None,
            fixed_last_step=fixed[-1]['step'] if fixed else None,brakes=sum(r['decision']['action']==8 for r in rs),navigation_blocked=sum(r['navigation_blocked'] for r in rs),
            contacts=e['capture_contact_step_count'],max_hold=e['capture_hold_counter_max'])
    capture.append(case)
write('capture_followup.json',capture)
for name,rows in [('assignment_changes.csv',changes),('low_speed_model.csv',slow),('partial_contexts.csv',contexts)]:
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in r.items()} for r in rows)
assert before==snap()
write('immutability.json',dict(passed=True,files_byte_unchanged=len(before),protected_counts={n:sum(k.startswith(n+'/') for k in before) for n in ('CH3','CH4','CH5')},environments=0,physical_steps=0,training_updates=0,checkpoint_restore=False))
write('created_files.json',sorted({p.relative_to(ROOT).as_posix() for p in OUT.iterdir() if p.is_file()}|{(OUT/'created_files.json').relative_to(ROOT).as_posix()}))
print(json.dumps(dict(probes_passed=len(probes),changes=len(changes),root_unchanged=sum(v['root_unchanged'] for v in changes),partial_events=len(contexts),partial_with_mismatch=sum(bool(v['other_mismatches']) for v in contexts),slow_desired_below_05=sum(v['desired_xy_inside_prior']<.05 for v in slow))))
