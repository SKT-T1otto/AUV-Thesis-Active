"""Offline brake algebra and true-box audit; never used for actual control."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from chapter3_bser.safe_rl.dynamics import integrate,brake,limits
from chapter3_bser.reach_avoid.options import SPECS
from chapter3_bser.safe_rl.options import DEFAULTS
from chapter3_bser.r123_task_sync.source import identity


def segment_box_gap(p,q,lo,hi):
    p,q,lo,hi=map(lambda x:np.array(x,float),(p,q,lo,hi));d=q-p;cuts=[0.,1.]
    for a in range(3):
        if abs(d[a])>1e-15:
            cuts.extend(t for t in ((lo[a]-p[a])/d[a],(hi[a]-p[a])/d[a]) if 0<t<1)
    cuts=sorted(set(cuts));values=[]
    for left,right in zip(cuts,cuts[1:]):
        mid=p+(left+right)/2*d;offset=np.zeros(3);slope=np.zeros(3)
        for a in range(3):
            if mid[a]<lo[a]:offset[a]=p[a]-lo[a];slope[a]=d[a]
            elif mid[a]>hi[a]:offset[a]=p[a]-hi[a];slope[a]=d[a]
        t=np.clip(-offset@slope/max(slope@slope,1e-30),left,right)
        for x in (left,right,t):values.append(float(np.linalg.norm(np.maximum(np.maximum(lo-(p+x*d),(p+x*d)-hi),0))))
    return min(values)


run=ROOT/'3090结果/r123_task_sync_20261118_223936_3078331'
done=json.loads((run/'sync/completed.json').read_text())
assert done['source']==identity()
assert hashlib.sha256((run/'sync/episodes.jsonl').read_bytes()).hexdigest()==done['artifacts']['episodes.jsonl']
raw=json.loads((run/'sync/episodes.jsonl').read_text(encoding='utf-8').splitlines()[0])
scene=json.loads((run/'plan/scenes.json').read_text())[0]
rs=[r for r in raw['path_option']['trace'] if r['agent']==0];spec=SPECS[0];dt=.2;timeline=[]
for t in range(138,154):
    r=rs[t];prev=rs[t-1];p=np.array(r['position']);v=np.array(r['velocity']);a0=np.array(prev['executed'])*limits(spec)
    flow=np.clip((v[:2]-(1-spec['drag_xy']*dt)*(np.array(prev['velocity'])[:2]+a0[:2]*dt))/dt,-.18,.18)
    command=brake(v,spec,dt,flow);predp,predv=integrate(p,v,command,spec,dt,flow)
    x,vel=p.copy(),v.copy();backup=[]
    for k in range(32):
        x,vel=integrate(x,vel,brake(vel,spec,dt,flow),spec,dt,flow)
        extra=float(np.sqrt(2)*2*.18*dt*dt*(k+1)*(k+2)/2)
        backup.append(dict(probe_step=k+1,position=x.tolist(),velocity=vel.tolist(),extra=extra))
        if k>=1 and np.linalg.norm(vel)<.04:break
    gap=min(segment_box_gap(p,r['next_position'],np.array(b['center'])-np.array(b['size'])/2,np.array(b['center'])+np.array(b['size'])/2) for b in scene['obstacles'])
    diag=r['action_diagnostics'];check=next(c for c in diag['candidate_checks'] if c['action']==8)
    item=dict(step=t,physical_step=t+1,position=r['position'],velocity=r['velocity'],next_position=r['next_position'],task=r['task'],tracking=r['event_target'],
        route_id=r['route_id'],map_revision=r['map_revision'],chosen_action=r['decision']['action'],emergency=r['emergency'],
        navigation_blocked=r['navigation_blocked'],verified_exit=diag['verified_exit'],cap_xy=diag['speed_cap_xy'],available_distance=diag['available_distance'],
        inferred_public_flow=flow.tolist(),brake_check=check,backup_under_fixed_public_flow=backup,offline_true_segment_clearance=gap,
        candidate_reasons={str(c['action']):c['reason'] for c in diag['candidate_checks']})
    if r['decision']['action']==8:
        item.update(brake_normalized_error=float(np.max(np.abs(command/limits(spec)-r['executed']))),
                    measured_next_velocity_error=float(np.max(np.abs(predv-rs[t+1]['velocity']))),
                    measured_next_position_error=float(np.max(np.abs(predp-r['next_position']))))
        assert item['brake_normalized_error']<1e-5
        assert abs(np.linalg.norm(predv[:2])-check['next_speed_xy'])<1e-6
    assert gap>0 and not r['collision']
    timeline.append(item)
result=dict(scene=raw['scenario_id'],agent=0,trace_indices_zero_based=True,environment_initializations=0,physical_steps=0,
    classification='Inherited emergency branch confirmed; no brake action/unit interface defect found; exact geometry cause REVIEW_REQUIRED',
    timeline=timeline,unknown=['Actual public hit/subcell/coarse occupancy snapshot at steps 144-145','Which public geometry component changed or tightened'],
    inference_scope='Flow reconstructed from preceding measured velocity and actual acceleration only; true boxes used offline solely for realized-segment clearance')
(OUT/'brake_analysis.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode())
print(json.dumps(dict(emergency_steps=[x['step'] for x in timeline if x['emergency']],brake_errors=[x['brake_normalized_error'] for x in timeline if 'brake_normalized_error' in x],min_true_clearance=min(x['offline_true_segment_clearance'] for x in timeline))))
