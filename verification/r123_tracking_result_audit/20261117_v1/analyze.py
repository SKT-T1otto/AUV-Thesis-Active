"""Read-only 2x2 tracking ablation audit; no envs, training or weight restores."""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from chapter3_bser.r123_tracking.source import identity,digest
from chapter3_bser.r123_tracking.runtime import ARMS
from chapter3_bser.r123_tracking.workflow import metric,aggregate
from chapter3_bser.r123_governed.source import identity as baseline_identity
from core.env.target_motion import TargetState,simulate_target_trajectory,swept_relative_min_distance

OUT=Path(__file__).resolve().parent
RUN=ROOT/'3090结果/r123_tracking_20261117_121047_2475513'
BASE=ROOT/'3090结果/r123_governed_20261116_215327_3078331'
ORDER=('baseline','path_only','feedforward_only','both')


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def legacy():
    return {folder:{p.relative_to(ROOT.parent/folder).as_posix():sha(p)
        for p in sorted((ROOT.parent/folder).rglob('*')) if p.is_file()
        and not any(x in p.parts for x in ('.git','__pycache__','.pytest_cache'))}
        for folder in ('CH3','CH4','CH5')}


def equal(a,b):
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if isinstance(a,(float,int)) and isinstance(b,(float,int)):return bool(np.isclose(a,b,rtol=1e-11,atol=1e-11))
    return a==b


def verify_done(folder,source):
    d=read(folder/'completed.json');assert d['source']==source
    for name in ('episodes','metrics','summary'):
        assert sha(folder/(name+('.json' if name=='summary' else '.jsonl')))==d[name+'_sha256'],(folder,name)
    return d


def distance(a,b):return float(np.linalg.norm(np.asarray(a)-b))
def search(raw):return [r for r in raw['path_option']['trace'] if r['agent']<3 and r['phase']==0]
PHYSICAL=('step','agent','phase','position','velocity','next_position','executed','event_target',
          'endpoint','reason','route_id','route_index','assignment_token')
def physical(raw,predicate=lambda r:True):
    return [{k:r[k] for k in PHYSICAL} for r in raw['path_option']['trace'] if predicate(r)]


manifest=read(ROOT/'provenance/r123_tracking_manifest_v1.json')
before=dict(source=identity(),legacy=legacy(),
    source_files={name:sha(ROOT/name) for name in manifest['files']},
    inputs={p.relative_to(ROOT).as_posix():sha(p) for base in (RUN,BASE)
            for p in sorted(base.rglob('*')) if p.is_file()})
write('before.json',before)
report=read(RUN/'report/report.json')
assert report['source']==before['source'] and report['complete_jobs']==report['expected_new_jobs']==3
v=read(RUN/'validation/validation.json')
assert v['source']==before['source'] and v['platform']=='Linux' and v['passed'] and v['native_smoke_passed']
assert v['environment_initializations']==4 and v['physical_steps']==128 and v['native_training_updates']==0
assert v['native_arms']==list(ORDER)
for name,h in v['artifacts'].items():assert sha(RUN/'validation'/name)==h,name
plan=read(RUN/'plan/plan.json')
assert plan['source']==before['source'] and plan['baseline_reused'] and plan['seed']==982001
for name,h in plan['inputs'].items():assert sha(RUN/'plan'/name)==h,name
config,opts=read(RUN/'plan/config.json'),read(RUN/'plan/options.json')
scenes=read(RUN/'plan/development.json')['scenarios'];ids=[s['scenario_id'] for s in scenes]
assert len(ids)==6 and config['max_steps']==400
assert config==read(BASE/'plan/config.json') and opts==read(BASE/'plan/options.json')
assert scenes==read(BASE/'plan/development.json')['scenarios']
done=verify_done(BASE/'governed',baseline_identity())
assert done['episodes_sha256']==read(RUN/'plan/baseline.json')['episodes_sha256']
raw={'baseline':rows(BASE/'governed/episodes.jsonl')}
metrics={'baseline':read(RUN/'plan/baseline.json')['rows']}
for arm in ORDER[1:]:
    folder=RUN/arm;d=verify_done(folder,before['source'])
    assert d['variant']==arm and d['switches']==dict(zip(('path','feedforward'),ARMS[arm]))
    assert d['seed']==982001 and d['scenario_ids']==ids
    assert d['config_hash']==digest(config) and d['options_hash']==digest(opts)
    raw[arm],metrics[arm]=rows(folder/'episodes.jsonl'),rows(folder/'metrics.jsonl')
    assert equal(read(folder/'summary.json'),aggregate(metrics[arm])),arm
for arm in ORDER:
    assert len(raw[arm])==len(metrics[arm])==6
    assert [e['scenario_id'] for e in raw[arm]]==ids
    assert equal(metrics[arm],report['arms'][arm]),arm
    for e,m in zip(raw[arm],metrics[arm]):
        assert all(equal(value,m[k]) for k,value in metric(e,400).items()),(arm,e['scenario_id'])
        assert len(e['path_option']['trace'])==4*e['terminal_step']
        assert not e['task_network_trained']
        assert all(not r['collision'] for r in e['path_option']['trace'])
        if arm!='baseline':
            assert e['path_option']['tracking_arm']==arm
            assert e['path_option']['tracking_switches']==dict(zip(('path','feedforward'),ARMS[arm]))
    assert sum(m['active_recovery_enters'] for m in metrics[arm])==0
for arm,delta in report['paired_differences'].items():
    for key,value in delta.items():
        actual=[n[key]-b[key] for b,n in zip(metrics['baseline'],metrics[arm])]
        assert equal(value,dict(per_scene=actual,mean=float(np.mean(actual)))),(arm,key)
for key,value in report['factorial_interaction_per_scene'].items():
    actual=[b[key]-p[key]-f[key]+o[key] for o,p,f,b in zip(*(metrics[a] for a in ORDER))]
    assert equal(actual,value),key

result=dict(source=before['source'],completed=True,complete_jobs=3,baseline_reused=True,
    linux_validation_tests=v['synthetic'],new_evaluation_physical_steps=sum(m['steps'] for a in ORDER[1:] for m in metrics[a]),
    audit_environment_initializations=0,audit_training_updates=0,arms={},paired_physical_checks={},cases=[])
SUM=('success','found','collision','timeout','steps','search_agent_steps','search_stationary_steps',
     'search_slow_distant_steps','search_brake_steps','route_continuity_advances','capture_contact_steps',
     'capture_fixed_waypoint_steps','active_recovery_enters')
for arm in ORDER:
    rr=raw[arm];mm=metrics[arm]
    total={key:sum(m[key] for m in mm) for key in SUM}
    total['slow_search_fraction']=total['search_slow_distant_steps']/total['search_agent_steps']
    total['conditional_capture_success']=total['success']/total['found']
    result['arms'][arm]=dict(totals=total,
        mean_discounted_return=float(np.mean([e['team_discounted_return'] for e in rr])),
        mean_undiscounted_return=float(np.mean([e['team_undiscounted_return'] for e in rr])))
predicates=dict(search=lambda r:r['agent']<3 and r['phase']==0,
                pre_capture=lambda r:r['phase'] in (0,1),capture=lambda r:r['agent']==3 and r['phase']==2)
for left,right,which in [('baseline','feedforward_only','search'),('path_only','both','search')]:
    matches=[physical(a,predicates[which])==physical(b,predicates[which]) for a,b in zip(raw[left],raw[right])]
    assert all(matches)
    result['paired_physical_checks'][f'{left}_vs_{right}_{which}']=matches
for left,right in [('baseline','path_only'),('feedforward_only','both')]:
    matches=[physical(a)==physical(b) for a,b in zip(raw[left],raw[right])]
    assert matches==[True,True,False,True,True,True]
    result['paired_physical_checks'][f'{left}_vs_{right}_all_agents']=matches

trajectories=[]
for i,s in enumerate(scenes):
    trajectory=simulate_target_trajectory(TargetState(np.asarray(s['target_initial_position'],np.float32),
        s['target_initial_velocity'],0,s['target_motion_mode']),400,.2,(20.,20.,8.),s['obstacles'],clearance=.2)
    trajectories.append(trajectory)
    case=dict(scenario_id=ids[i],arms={})
    for arm in ORDER:
        e=raw[arm][i];trace=e['path_option']['trace'];observations=[]
        for r in search(e):
            t=r['step'];d,_=swept_relative_min_distance(r['position'],r['next_position'],trajectory[t].position,trajectory[t+1].position)
            observations.append(dict(step=t+1,agent=r['agent'],distance=d,margin=d-(2.1,2.45,2.85)[r['agent']]))
        detected=[o['step'] for o in observations if o['margin']<=0]
        assert (min(detected) if detected else None)==e['found_step'],(arm,i)
        capture=[r for r in trace if r['agent']==3 and r['phase']==2]
        contacts=hold=maximum=0;distances=[]
        for r in capture:
            t=r['step'];d,_=swept_relative_min_distance(r['position'],r['next_position'],trajectory[t].position,trajectory[t+1].position)
            full=max(distance(r['position'],trajectory[t].position),distance(r['next_position'],trajectory[t+1].position))<=.8
            contacts+=int(d<=.8);hold=hold+1 if full else 0;maximum=max(maximum,hold);distances.append((d,t+1))
        assert contacts==e['capture_contact_step_count'] and maximum==e['capture_hold_counter_max'],(arm,i)
        value=dict(found_step=e['found_step'],received_step=e['executor_target_received_step'],
            success_step=e['success_step'],found_to_success=e['found_to_success_steps'],
            contacts=contacts,hold_max=maximum,closest_search=min(observations,key=lambda x:x['margin']),
            closest_capture=min(distances) if distances else None,
            capture_brakes=sum(r['decision']['action']==8 for r in capture))
        if i==2:
            agent=[r for r in trace if r['agent']==1]
            back=[y['step'] for x,y in zip(agent,agent[1:]) if x['event_target']==[5.,5.,2.5] and y['event_target']==[3.,3.,2.5]]
            value.update(agent1_slow_steps=sum(np.linalg.norm(r['velocity'])<.05 and distance(r['position'],r['endpoint'])>.8 for r in agent),
                backwards_5_to_3_steps=back,backwards_in_100_399=sum(t>=100 for t in back),
                final_position=agent[-1]['next_position'],
                continuity_events=e['path_option'].get('route_continuity_events',[]))
        case['arms'][arm]={k:v.item() if isinstance(v,np.generic) else v for k,v in value.items()}
    result['cases'].append(case)
write('analysis.json',result)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
labels=('baseline','path only','feedforward only','both')
colors=('#64748b','#f59e0b','#2563eb','#16a34a')
ax=axs[0,0]
for n,arm in enumerate(ORDER):
    count=result['arms'][arm]['totals']['success'];ax.bar(n,count,color=colors[n]);ax.text(n,count+.06,f'{count}/6',ha='center')
ax.set(xticks=range(4),xticklabels=labels,ylim=(0,6),ylabel='Successful episodes',title='Mission success: same six scenes')
ax=axs[0,1]
for n,arm in enumerate(ORDER):
    count=result['arms'][arm]['totals']['search_slow_distant_steps'];ax.bar(n,count,color=colors[n]);ax.text(n,count+6,str(count),ha='center')
ax.set(xticks=range(4),xticklabels=labels,ylim=(0,520),ylabel='Slow distant search-agent steps',title='Path repair reduces low-speed search')
ax=axs[1,0]
for arm,color in (('baseline',colors[0]),('path_only',colors[1])):
    r=[r for r in raw[arm][2]['path_option']['trace'] if r['agent']==1]
    p=np.array([x['next_position'] for x in r]);ax.plot(p[:,0],p[:,1],label=arm,color=color,lw=1.8)
    ax.scatter(*p[-1,:2],color=color,s=55,zorder=5)
ax.set(title='Case 0002, agent 1: actual XY trajectory',xlabel='X (m)',ylabel='Y (m)');ax.set_aspect('equal',adjustable='datalim');ax.legend()
ax=axs[1,1]
for arm,color in (('baseline',colors[0]),('feedforward_only',colors[2])):
    rs=[r for r in raw[arm][4]['path_option']['trace'] if r['agent']==3 and r['phase']==2]
    ax.plot([r['step']+1 for r in rs],[distance(r['next_position'],trajectories[4][r['step']+1].position) for r in rs],color=color,label=arm,lw=1.8)
ax.axhline(.8,color='black',ls='--',lw=1,label='capture radius')
ax.set(title='Case 0004: executor-to-target distance',xlabel='Physical step',ylabel='Distance (m)');ax.legend()
fig.savefig(OUT/'ablation.png',dpi=160)
fig.savefig(OUT/'ablation.svg');plt.close(fig)

assert legacy()==before['legacy']
assert all(sha(ROOT/name)==h for name,h in before['inputs'].items())
assert all(sha(ROOT/name)==h for name,h in before['source_files'].items())
assert identity()==before['source']
write('immutability.json',dict(source_files_unchanged=len(before['source_files']),input_files_unchanged=len(before['inputs']),
    protected_counts={k:len(v) for k,v in before['legacy'].items()},protected_byte_identical=True,
    environment_initializations=0,training_updates=0,checkpoint_restore=False))
write('created_files.json',sorted({p.relative_to(ROOT).as_posix() for p in OUT.iterdir() if p.is_file()}|
                                {(OUT/'created_files.json').relative_to(ROOT).as_posix()}))
print(json.dumps(dict(completed=True,new_evaluation_steps=result['new_evaluation_physical_steps'],
    success={a:result['arms'][a]['totals']['success'] for a in ORDER},verified_raw_episodes=24,output=str(OUT)),ensure_ascii=False))
