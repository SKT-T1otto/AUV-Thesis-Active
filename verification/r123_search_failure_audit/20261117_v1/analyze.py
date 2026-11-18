"""Offline trace audit. No environments, controller rollouts, training or restores."""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from core.env.target_motion import TargetState, simulate_target_trajectory, swept_relative_min_distance

RUN = ROOT / '3090结果/r123_tracking_20261117_121047_2475513'
BASE = ROOT / '3090结果/r123_governed_20261116_215327_3078331'

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
def scalar(x):
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x).__name__)
def write(name, data):
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False,default=scalar)+'\n', encoding='utf-8')
def tree(base): return {p.relative_to(base).as_posix(): sha(p) for p in sorted(base.rglob('*')) if p.is_file()}
def snapshot():
    return dict(protected={n:tree(ROOT.parent/n) for n in ('CH3','CH4','CH5')},
                inputs={p.relative_to(ROOT).as_posix():sha(p) for b in (RUN,BASE,ROOT/'verification/r123_tracking_result_audit/20261117_v1') for p in b.rglob('*') if p.is_file()},
                source={p.relative_to(ROOT).as_posix():sha(p) for b in ('core','chapter3_bser','tools','provenance') for p in (ROOT/b).rglob('*') if p.is_file() and '__pycache__' not in p.parts})
def counter(values): return dict(Counter(values))
def norm(x,y): return float(np.linalg.norm(np.asarray(x)-y))
def truth_gap(p,scene):
    p=np.asarray(p); boxes=scene['obstacles']
    return min(float(np.linalg.norm(np.maximum(np.abs(p-np.asarray(b['center']))-np.asarray(b['size'])/2,0))) for b in boxes)
def intervals(rows):
    groups=[]
    for r in rows:
        if not groups or r['step'] != groups[-1][-1]['step']+1: groups.append([])
        groups[-1].append(r)
    return groups
def stats(v): return dict(min=float(min(v)),median=float(np.median(v)),max=float(max(v)))

before = read(OUT/'before.json') if (OUT/'before.json').exists() else snapshot()
assert before==snapshot()
if not (OUT/'before.json').exists(): write('before.json',before)
scenes = read(RUN/'plan/development.json')['scenarios']
episodes = [json.loads(x) for x in (RUN/'both/episodes.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(episodes)==len(scenes)==6
result = dict(environment_initializations=0, physical_steps=0, training_updates=0, cases=[], slow_total=0,
              definitions=dict(step='trace step is zero based; interval [step*.2,(step+1)*.2]; event step uses same counter',
              slow='pre-step speed < .05 and pre-step endpoint distance > .8; search agents phase 0',
              visit='sampled pre-step position voxel (2m,2m,1m); entry counts exclude consecutive dwell; not sensor/map coverage'))
slow_csv=[]; task_csv=[]; trajectories=[]
for si,(scene,e) in enumerate(zip(scenes,episodes)):
    assert scene['scenario_id']==e['scenario_id']
    trajectory=simulate_target_trajectory(TargetState(np.asarray(scene['target_initial_position'],np.float32),scene['target_initial_velocity'],0,scene['target_motion_mode']),400,.2,(20.,20.,8.),scene['obstacles'],clearance=.2)
    tp=np.array([s.position for s in trajectory]);trajectories.append(tp)
    trace=e['path_option']['trace'];ev=e['path_option']['lifecycle_events']
    case=dict(scenario_id=e['scenario_id'],found_step=e['found_step'],planner=e['planner'],events=ev,agents=[],
              search_endpoint_duplicate_steps=0,root_endpoint_duplicate_steps=0)
    case['target_position_min']=tp.min(axis=0).tolist();case['target_position_max']=tp.max(axis=0).tolist()
    case['task_complete_checks']=[]
    for v in ev:
        if v['event']=='TASK_RELEASE' and v['reason']=='TASK_COMPLETE' and v['agent']<3:
            prior=next(r for r in trace if r['agent']==v['agent'] and r['step']==v['step']-1)
            d=norm(prior['next_position'],prior['task'])
            case['task_complete_checks'].append(dict(agent=v['agent'],step=v['step'],root_distance=d,within_arrival_radius=d<=.750001))
    detect=[]
    for t in range(e['found_step'] or e['terminal_step']):
        rr=[r for r in trace if r['step']==t and r['agent']<3 and r['phase']==0]
        for key,label in [('original','search_endpoint_duplicate_steps'),('task','root_endpoint_duplicate_steps')]:
            case[label]+=int(len({tuple(r[key]) for r in rr})<len(rr))
    for a in range(3):
        rs=[r for r in trace if r['agent']==a and r['phase']==0]
        ps=np.array([r['position'] for r in rs]); cells=np.floor(ps/[2,2,1]).astype(int)
        entries=[tuple(c) for n,c in enumerate(cells) if n==0 or not np.array_equal(c,cells[n-1])]
        visits=Counter(entries); radius=(2.1,2.45,2.85)[a]
        distances=[]
        for r in rs:
            t=r['step'];d,tau=swept_relative_min_distance(r['position'],r['next_position'],tp[t],tp[t+1]);detect.append(t+1 if d<=radius else None)
            distances.append(dict(step=t,physical_step=t+1,time_seconds=(t+tau)*.2,distance=d,margin=d-radius,tau=tau,position=r['position'],target_start=tp[t].tolist()))
        closest=min(distances,key=lambda x:x['distance'])
        spatial=np.linalg.norm(ps[:,None,:]-tp[None,:,:],axis=2);ix,tx=np.unravel_index(np.argmin(spatial),spatial.shape)
        slow=[r for r in rs if np.linalg.norm(r['velocity'])<.05 and norm(r['position'],r['endpoint'])>.8]
        runs=[]
        for group in intervals(slow):
            runs.append(dict(start=group[0]['step'],end=group[-1]['step'],count=len(group),
                reasons=counter(r['reason'] for r in group),route_ids=sorted({r['route_id'] for r in group}),
                tracking_points=list({tuple(r['event_target']) for r in group}),endpoints=list({tuple(r['endpoint']) for r in group}),
                cap_xy=stats([r['action_diagnostics']['speed_cap_xy'] for r in group]),
                offline_true_aabb_gap=stats([truth_gap(r['position'],scene) for r in group]),
                tracking_distance=stats([r['distance_to_tracking'] for r in group]),
                navigation_blocked=sum(r['navigation_blocked'] for r in group),brakes=sum(r['decision']['action']==8 for r in group)))
        for r in slow:
            d=r['action_diagnostics']
            assert d['deployed_action_mask'][r['decision']['action']]
            slow_csv.append(dict(scene=si,agent=a,step=r['step'],reason=r['reason'],speed=float(np.linalg.norm(r['velocity'])),
                position=r['position'],task=r['task'],endpoint=r['endpoint'],tracking=r['event_target'],route_id=r['route_id'],route_index=r['route_index'],
                endpoint_distance=norm(r['position'],r['endpoint']),tracking_distance=r['distance_to_tracking'],cap_xy=d['speed_cap_xy'],cap_z=d['speed_cap_z'],
                available_distance=d['available_distance'],navigation_blocked=r['navigation_blocked'],action=r['decision']['action'],
                offline_true_aabb_gap=truth_gap(r['position'],scene),
                candidate_reasons=counter(x['reason'] for x in d['candidate_checks']),deployed_mask=d['deployed_action_mask']))
        mismatch=[r for r in rs if norm(r['original'],r['task'])>1e-4]
        accepts=[v for v in ev if v.get('agent')==a and v['event']=='TASK_ACCEPT']
        ledger=[]
        for n,event in enumerate(accepts):
            end=accepts[n+1]['step'] if n+1<len(accepts) else len(rs)
            segment=[r for r in rs if event['step']<=r['step']<end]
            if not segment: continue
            releases=[v for v in ev if v.get('task_id')==event['task_id'] and v['event']=='TASK_RELEASE']
            item=dict(scene=si,agent=a,task_id=event['task_id'],start=event['step'],end=end-1,root=segment[0]['task'],
                      release=releases,original_mismatch_steps=sum(norm(r['original'],r['task'])>1e-4 for r in segment),
                      min_root_distance=min(norm(r['position'],segment[0]['task']) for r in segment),last_position=segment[-1]['next_position'])
            ledger.append(item); task_csv.append(item)
        agent=dict(agent=a,radius=radius,closest=closest,slow_count=len(slow),slow_intervals=runs,
            reasons=counter(r['reason'] for r in rs),actions=counter(str(r['decision']['action']) for r in rs),
            path_length=sum(norm(r['position'],r['next_position']) for r in rs),
            position_min=ps.min(axis=0).tolist(),position_max=ps.max(axis=0).tolist(),
            unique_position_voxels=len(visits),voxel_entries=len(entries),repeat_voxel_entries=len(entries)-len(visits),
            most_entered_voxels=[dict(cell=k,count=v) for k,v in visits.most_common(5)],
            root_original_mismatch_steps=len(mismatch),mismatch_intervals=[[g[0]['step'],g[-1]['step']] for g in intervals(mismatch)],
            task_endpoint_mismatch_steps=sum(norm(r['task'],r['endpoint'])>1e-4 for r in rs),
            sampled_spatial_closest=dict(distance=float(spatial[ix,tx]),search_step=rs[ix]['step'],target_step=int(tx),within_radius=bool(spatial[ix,tx]<=radius)),
            task_ledger=ledger)
        case['agents'].append(agent);result['slow_total']+=len(slow)
    assert min((x for x in detect if x is not None),default=None)==e['found_step']
    case['slow_count']=sum(a['slow_count'] for a in case['agents']);result['cases'].append(case)
assert result['slow_total']==319
write('analysis.json',result)
for name,data in [('slow_steps.csv',slow_csv),('tasks.csv',task_csv)]:
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader()
        w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()} for r in data)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(3,2,figsize=(13,13),constrained_layout=True)
for si in range(3):
    ax,dx=axs[si];scene=scenes[si];tp=trajectories[si]
    ax.plot(tp[:,0],tp[:,1],color='black',ls='--',label='target (offline)')
    for b in scene['obstacles']:
        c=np.array(b['center']);s=np.array(b['size']);ax.add_patch(plt.Rectangle(c[:2]-s[:2]/2,*s[:2],color='gray',alpha=.25))
    for a in range(3):
        rs=[r for r in episodes[si]['path_option']['trace'] if r['agent']==a];ps=np.array([r['position'] for r in rs]);color=f'C{a}'
        ax.plot(ps[:,0],ps[:,1],color=color,label=f'agent {a}');ax.scatter(*ps[0,:2],color=color,marker='o');ax.scatter(*ps[-1,:2],color=color,marker='x')
        margins=[swept_relative_min_distance(r['position'],r['next_position'],tp[r['step']],tp[r['step']+1])[0]-(2.1,2.45,2.85)[a] for r in rs]
        dx.plot([(r['step']+1)*.2 for r in rs],margins,color=color,label=f'agent {a}')
    ax.set(title=f'Case {si:04d} XY projection (not coverage)',xlim=(0,20),ylim=(0,20),xlabel='X (m)',ylabel='Y (m)');ax.set_aspect('equal');ax.legend(fontsize=8)
    dx.axhline(0,color='black',ls='--');dx.set(title='Synchronous swept distance minus detection radius',xlabel='Time (s)',ylabel='Margin (m)');dx.legend(fontsize=8)
fig.savefig(OUT/'failure_paths.png',dpi=150);plt.close(fig)
after=snapshot();assert before==after,'Input/source/protected file bytes changed'
write('immutability.json',dict(protected_byte_identical=True,protected_counts={n:len(v) for n,v in before['protected'].items()},source_files_unchanged=len(before['source']),input_files_unchanged=len(before['inputs']),environment_initializations=0,physical_steps=0,training_updates=0,checkpoint_restore=False))
write('created_files.json',sorted({p.relative_to(ROOT).as_posix() for p in OUT.iterdir() if p.is_file()}|{(OUT/'created_files.json').relative_to(ROOT).as_posix()}))
print(json.dumps(dict(slow_total=result['slow_total'],cases=[dict(scene=i,slow=c['slow_count'],closest=[a['closest']['distance'] for a in c['agents']]) for i,c in enumerate(result['cases'])]),ensure_ascii=False))
