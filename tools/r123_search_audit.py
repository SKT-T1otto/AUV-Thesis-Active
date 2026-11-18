"""Portable offline audit of existing tracking results; zero environment steps."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from project_paths import ROOT
from chapter3_bser.r123_tracking.source import identity
from chapter3_bser.r123_governed.source import identity as governed_identity
from core.env.target_motion import TargetState,simulate_target_trajectory,swept_relative_min_distance
from chapter3_bser.reach_avoid.options import SPECS,CONTRACT
from chapter3_bser.safe_rl.options import DEFAULTS as SAFETY
from chapter3_bser.safe_rl.dynamics import speed_limit


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):return [json.loads(v) for v in p.read_text(encoding='utf-8').splitlines() if v]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def require(condition,message):
    if not condition:raise ValueError(message)
def under(a,b):return a==b or b in a.parents
def hashes(bases):return {str(p):sha(p) for b in bases for p in sorted(b.rglob('*')) if p.is_file()}


def audit(run,baseline,out):
    run,baseline,out=(p.resolve() for p in (run,baseline,out))
    require(not out.exists(),'output must be a new directory')
    require(not any(under(out,p) or under(p,out) for p in (run,baseline)),'output must be separate from inputs')
    require((run/'plan/plan.json').is_file(),'tracking plan missing')
    require((baseline/'completed.json').is_file(),'governed baseline completion missing')
    source=identity();plan=read(run/'plan/plan.json');before=hashes((run,baseline))
    require(plan['source']==source,'plan/source mismatch')
    for n,h in plan['inputs'].items():require(sha(run/'plan'/n)==h,'plan input mismatch: '+n)
    scenes=read(run/'plan/development.json')['scenarios']
    config=read(run/'plan/config.json')
    require(len(scenes)==6 and config['max_steps']==400,'expected original six 400-step scenes')
    require(plan['baseline_reused'] and not plan['training_allowed'],'unexpected experiment protocol')
    folders={'baseline':baseline,**{a:run/a for a in ('path_only','feedforward_only','both')}}
    data={}
    for arm,folder in folders.items():
        done=read(folder/'completed.json')
        require(done['source']==(governed_identity() if arm=='baseline' else source),'completion/source mismatch: '+arm)
        for name,ext in (('episodes','.jsonl'),('metrics','.jsonl'),('summary','.json')):
            require(sha(folder/(name+ext))==done[name+'_sha256'],'completion hash mismatch: '+arm+'/'+name)
        data[arm]=rows(folder/'episodes.jsonl')
        require(len(data[arm])==6,'wrong episode count: '+arm)
    require(sha(baseline/'episodes.jsonl')==read(run/'plan/baseline.json')['episodes_sha256'],'wrong imported baseline')
    tool_hash=sha(Path(__file__))
    out.mkdir(parents=True,exist_ok=False)
    write(out/'request.json',dict(source=source,tool_sha256=tool_hash,run=str(run),baseline=str(baseline),
        environment_initializations=0,physical_steps=0,training=False,checkpoint_load=False))
    try:
        result=dict(source=source,tool_sha256=tool_hash,arms={})
        trajectories=[simulate_target_trajectory(TargetState(np.asarray(s['target_initial_position'],np.float32),s['target_initial_velocity'],0,s['target_motion_mode']),400,.2,(20.,20.,8.),s['obstacles'],clearance=.2) for s in scenes]
        for arm,episodes in data.items():
            cases=[]
            for n,e in enumerate(episodes):
                require(e['scenario_id']==scenes[n]['scenario_id'],'scenario order mismatch')
                require(not e['task_network_trained'],'unexpected trained episode')
                trace=e['path_option']['trace'];detected=[];agents=[];slow=desiredslow=0
                for a in range(3):
                    rr=[r for r in trace if r['agent']==a and r['phase']==0];closest=None;agent_slow=0
                    for r in rr:
                        t=r['step'];tp=trajectories[n]
                        d,tau=swept_relative_min_distance(r['position'],r['next_position'],tp[t].position,tp[t+1].position)
                        margin=d-(2.1,2.45,2.85)[a]
                        item=dict(step=t,physical_step=t+1,time_seconds=(t+tau)*.2,distance=d,margin=margin)
                        if closest is None or margin<closest['margin']:closest=item
                        if margin<=0:detected.append(t+1)
                        p=np.array(r['position']);v=np.array(r['velocity'])
                        if np.linalg.norm(v)<.05 and np.linalg.norm(p-r['endpoint'])>.8:
                            slow+=1;agent_slow+=1;diag=r['action_diagnostics']
                            cap=speed_limit(diag['available_distance'],SPECS[a],CONTRACT['dt'],SAFETY['flow_component_bound'],reaction_steps=SAFETY['reaction_steps'])
                            require(abs(cap-diag['speed_cap_xy'])<1e-12,'cap reconstruction mismatch')
                            desired=cap*min(float(np.linalg.norm((np.array(r['event_target'])-p)[:2]))/CONTRACT['prior_slow_radius_xy'],1.)
                            desiredslow+=int(desired<.05)
                    agents.append(dict(agent=a,closest=closest,slow_steps=agent_slow,
                        assigned_root_mismatch_steps=sum(r['assignment_kind']=='search' and r['original']!=r['task'] for r in rr)))
                require(min(detected,default=None)==e['found_step'],'discovery reconstruction mismatch')
                require(not any(v['event']=='RECOVERY_ENTER' for v in e['path_option']['lifecycle_events']),'unexpected active recovery')
                cases.append(dict(scenario_id=e['scenario_id'],success=bool(e['success']),found=bool(e['found']),
                    collision=bool(e['collision_episode']),slow_steps=slow,prior_desired_xy_below_05=desiredslow,agents=agents))
                print(f'[offline audit] arm={arm} scene={n+1}/6 physical_steps=0',flush=True)
            result['arms'][arm]=dict(success=sum(c['success'] for c in cases),found=sum(c['found'] for c in cases),
                collision=sum(c['collision'] for c in cases),slow_steps=sum(c['slow_steps'] for c in cases),cases=cases)
        require(result['arms']['both']['slow_steps']==319,'expected supplied both slow total')
        require(before==hashes((run,baseline)),'input bytes changed during audit')
        require(source==identity() and tool_hash==sha(Path(__file__)),'source/tool changed during audit')
        write(out/'analysis.json',result)
        write(out/'immutability.json',dict(inputs_unchanged=True,input_hashes=before,source_unchanged=True,
            environment_initializations=0,physical_steps=0,training_updates=0,checkpoint_restore=False))
        report=['# Existing tracking results: offline recheck','','No environment initialization, rollout, training or checkpoint load.','',
            '| arm | success | found | collisions | slow search steps |','|---|---:|---:|---:|---:|']
        for arm,m in result['arms'].items():report.append(f'| {arm} | {m["success"]}/6 | {m["found"]}/6 | {m["collision"]} | {m["slow_steps"]} |')
        (out/'REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
        artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()}
        write(out/'completed.json',dict(completed=True,source=source,tool_sha256=tool_hash,verified_episodes=24,
            physical_steps=0,training=False,artifacts=artifacts,created_files=sorted([*artifacts,'completed.json'])))
        print(json.dumps(dict(completed=True,output=str(out),physical_steps=0),ensure_ascii=False))
    except Exception as exc:
        write(out/'failed.json',dict(error=str(exc),completed=False,physical_steps=0));raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.execute:audit(a.run,a.baseline,a.output)
    else:print(json.dumps(dict(executed=False,run=str(a.run),baseline=str(a.baseline),output=str(a.output),physical_steps=0)))


if __name__=='__main__':main()
