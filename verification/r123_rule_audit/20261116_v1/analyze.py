"""Read-only analysis of supplied Linux rollouts; no simulator or weights."""
import collections
import csv
import hashlib
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
OUT=Path(__file__).resolve().parent
DATA=ROOT/'3090结果/r123_20261116_103339_2030716'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,value):
    encoded=json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False,
        default=lambda value:value.item() if isinstance(value,np.generic) else str(value))
    with (OUT/name).open('x',encoding='utf-8') as f:f.write(encoded)
def dist(a,b):return float(np.linalg.norm(np.asarray(a)-b))
def runs(rows,predicate):
    result=[];start=None
    for index,row in enumerate(rows):
        if predicate(row) and start is None:start=index
        if start is not None and (not predicate(row) or index==len(rows)-1):
            end=index if not predicate(row) else index+1
            result.append(dict(start=rows[start]['step'],end=rows[end-1]['step']+1,length=end-start))
            start=None
    return result

def main():
    from chapter3_bser.r123.source import identity
    from chapter3_bser.r123.lifecycle import Lifecycle,ProgressManager
    from chapter3_bser.r123.options import CONTROL
    from chapter3_bser.path_option.options import DEFAULTS
    from chapter3_bser.path_option.path_context import Route
    from chapter3_bser.d2.allocator import EffectAllocator
    gate=read(DATA/'rule/reference_gate.json');evaluation=read(DATA/'rule/evaluation.json')
    assert gate['source']==identity()==evaluation['source']
    assert gate['summary_hash']==sha(DATA/'rule/summary.json')
    assert gate['evaluation_hash']==sha(DATA/'rule/evaluation.json')
    assert not gate['passed']
    receipt=read(DATA/'validation/validation.json')
    assert receipt['passed'] and receipt['platform']=='Linux' and receipt['source']==identity()
    for name,digest in receipt['artifacts'].items():assert sha(DATA/'validation'/name)==digest
    archive=[]
    with tarfile.open(DATA/'rule_diagnostics.tar.gz','r:gz') as t:
        for member in t.getmembers():
            if not member.isfile():continue
            contents=t.extractfile(member).read()
            assert hashlib.sha256(contents).hexdigest()==sha(DATA/member.name)
            archive.append(member.name)
    raw=lines(DATA/'rule/episodes.jsonl');metrics=lines(DATA/'rule/metrics.jsonl')
    assert len(raw)==len(metrics)==6
    cases=[];recoveries=[];agents=[];placeholder=[];post_found=[]
    for case,(episode,metric) in enumerate(zip(raw,metrics)):
        trace=episode['path_option']['trace'];events=episode['path_option']['lifecycle_events']
        assert metric['scenario_id']==episode['scenario_id']
        assert len(trace)==4*episode['actual_length']
        phases={e['task_id']:e['phase'] for e in events if e['event']=='TASK_ACCEPT'}
        index={(r['step'],r['agent']):r for r in trace}
        for e in events:
            if e['event']=='TASK_ACCEPT' and e['phase']==0 and (e['step'],e['agent']) in index:
                row=index[e['step'],e['agent']]
                if dist(row['task'],row['position'])<1e-5:
                    placeholder.append(dict(case=case,**e,task=row['task'],same_step_release=any(
                        x['event']=='TASK_RELEASE' and x['task_id']==e['task_id'] and x['step']==e['step'] for x in events)))
        for taskid in sorted(phases):
            ent=[e for e in events if e['event']=='RECOVERY_ENTER' and e['task_id']==taskid]
            rel=[e for e in events if e['event']=='TASK_RELEASE' and e['task_id']==taskid]
            if not ent:continue
            stop=rel[0]['step'] if rel else None
            group=dict(case=case,task_id=taskid,phase=phases[taskid],agent=ent[0]['agent'],
                enters=[e['step'] for e in ent],release_step=stop,release_reason=rel[0]['reason'] if rel else None,
                first_to_release=None if stop is None else stop-ent[0]['step'])
            recoveries.append(group)
        for a in range(4):
            allrows=[r for r in trace if r['agent']==a]
            active=[r for r in allrows if r['phase']!=3]
            waits=runs(active,lambda r:r['reason']=='WAIT_TASK')
            frozen=[r for r in allrows if r['phase']==3]
            if frozen:
                post_found.append(dict(case=case,agent=a,steps=len(frozen),
                    option_terminations=sum(r['decision']['terminate'] for r in frozen),
                    recovery_enters=sum(e['event']=='RECOVERY_ENTER' and e.get('agent')==a and phases.get(e.get('task_id'))==3 for e in events),
                    nonzero_actions=sum(np.linalg.norm(r['executed'])>1e-7 for r in frozen),
                    distance=sum(dist(r['next_position'],r['position']) for r in frozen)))
            agents.append(dict(case=case,agent=a,active_steps=len(active),
                reasons=dict(collections.Counter(r['reason'] for r in active)),wait_runs=waits,
                wait_steps=sum(w['length'] for w in waits),
                forced=dict(collections.Counter(str(r['forced']) for r in active)),
                distance=sum(dist(r['next_position'],r['position']) for r in active),
                final_task=allrows[-1]['task'],final_original=allrows[-1]['original'],final_reason=allrows[-1]['reason']))
        search=[r for r in trace if r['agent']<3 and r['phase']==0]
        wait=sum(r['reason']=='WAIT_TASK' for r in search)
        final_wait_starts=[]
        for a in range(3):
            rr=[r for r in trace if r['agent']==a]
            tail=runs(rr,lambda r:r['reason']=='WAIT_TASK')
            final_wait_starts.append(tail[-1]['start'] if tail and tail[-1]['end']==episode['actual_length'] else None)
        d2=[e for e in events if e['event']=='D2_RELEASE_REQUEST']
        safe=episode['planner']['safe_counts']
        cases.append(dict(case=case,scenario_id=episode['scenario_id'],success=episode['success'],found=episode['found'],
            found_step=episode['found_step'],handoff_step=episode['executor_target_received_step'],success_step=episode['success_step'],
            steps=episode['actual_length'],search_agent_steps=len(search),wait_steps=wait,wait_fraction=wait/max(1,len(search)),
            terminal_all_searchers_wait_steps=episode['actual_length']-max(final_wait_starts) if all(x is not None for x in final_wait_starts) else 0,
            recovery_attempts=metric['recovery_attempts'],recovery_returns=metric['recovery_returns'],
            d2_requests=len(d2),d2_rejected=sum(not e['accepted'] for e in d2),
            d2_reasons=dict(collections.Counter(e['reason'] for e in d2)),
            safe_counts=safe,motion_distance=metric['motion_distance'],capture_hold_max=metric['capture_hold_max']))
    # Isolated deterministic probes of existing production classes. No env.
    life=Lifecycle(CONTROL);life.accept(0,[9.,3.,4.],0,0,[3.,3.,4.])
    manager=ProgressManager(DEFAULTS,life,True);manager.public_pose[0]=(True,2.)
    manager.commit(0,[9.,3.,4.],[9.,3.,4.],[9.,3.,4.],0,0,Route([[3.,3.,4.],[9.,3.,4.]],1,0,True),reason='INITIAL')
    stalled=manager.status(0,np.array([3.,3.,4.]),np.zeros(3),0,30,True)
    life.enter_recovery(0,30,[5.,5.,4.])
    manager.commit(0,[9.,3.,4.],[9.,3.,4.],[5.,5.,4.],30,0,Route([[3.,3.,4.],[5.,5.,4.]],1,30,True),reason='STALLED')
    immediate=manager.status(0,np.array([3.,3.,4.]),np.zeros(3),0,31,True)
    assert stalled==immediate=='STALLED'
    allocator=EffectAllocator.__new__(EffectAllocator)
    allocator.search_pool=lambda state,ids:([NS(agent_id=0)],0,())
    _,accepted,reason=allocator.allocate_partial(NS(target_found=False,searcher_ids=(0,1)),object(),affected_searcher_ids=(0,1),trigger_reason='SYNTHETIC_AUDIT')
    assert not accepted and reason=='ATOMIC_REJECT_MISSING_SEARCH_ROUTE'
    active_recoveries=[r for r in recoveries if r['phase'] in (0,1,2)]
    released=[r for r in active_recoveries if r['first_to_release'] is not None]
    totals=dict(failed_cases=sum(not c['success'] for c in cases),
        failed_search_wait_steps=sum(c['wait_steps'] for c in cases if not c['success']),
        failed_search_agent_steps=sum(c['search_agent_steps'] for c in cases if not c['success']),
        active_recovery_enters=sum(len(r['enters']) for r in active_recoveries),
        active_recovery_tasks=len(active_recoveries),released_recovery_tasks=len(released),
        released_within_3_steps=sum(r['first_to_release']<=3 for r in released),
        recovery_return_count=sum(c['recovery_returns'] for c in cases),
        frozen_recovery_enters=sum(r['recovery_enters'] for r in post_found),
        placeholder_task_accepts=len(placeholder),placeholder_immediate_releases=sum(x['same_step_release'] for x in placeholder))
    write('analysis_v2.json',dict(source=identity(),archive_verified=archive,validation_artifacts_verified=True,
        cases=cases,agents=agents,recovery_task_timelines=recoveries,placeholder_tasks=placeholder,
        frozen_searcher_bookkeeping=post_found,totals=totals,
        synthetic_probes=dict(environment_initializations=0,training_updates=0,checkpoint_loads=0,
            fresh_recovery_reason=immediate,fresh_recovery_age=1,root_progress_step=life.tasks[0].progress_step,
            group_allocation_with_one_missing_pool=dict(accepted=accepted,reason=reason)),
        limits=['No causal rerun or reward/algorithm change','No public map snapshots in supplied traces; per-agent feasibility after atomic rejection remains REVIEW_REQUIRED',
                'No trained R123 policy exists in this run; no RL effect can be measured']))
    with (OUT/'cases.csv').open('x',encoding='utf-8-sig',newline='') as f:
        columns=['case','scenario_id','success','found','steps','wait_steps','wait_fraction','terminal_all_searchers_wait_steps','recovery_attempts','recovery_returns','d2_requests','d2_rejected']
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(cases)
    print(json.dumps(totals,indent=2));print(json.dumps([dict(case=c['case'],wait_fraction=c['wait_fraction'],d2_rejected=c['d2_rejected'],d2_requests=c['d2_requests']) for c in cases],indent=2))

if __name__=='__main__':main()
