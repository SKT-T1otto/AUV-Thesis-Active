"""Public-data endurance and fault checks; zero simulator episodes or models."""
import argparse
import copy
import csv
import json
from pathlib import Path
import time
import numpy as np
from project_paths import ROOT
from core.env.public_events import PublicEvent,StepObservation,payload_digest,vectors
from chapter3_bser.experiments.coupled_search_v1.types import PublicStepEvidence
from chapter3_bser.experiments.coupled_search_v1.bounded_belief import BoundedCoupledBelief
from chapter3_bser.experiments.coupled_search_v1.coupled_belief import CoupledBelief
from tools.check_map_cache import synthetic_history,CONTRACT
from checks.test_coupled_search import contract,specs
from chapter3_bser.learning.bounded_experiment import timing_bands
from chapter3_bser.learning.checkpoint import digest,write_json


def packets(count):
    result=[];agents=vectors([[18.,18.,9.]]*4);velocity=vectors(np.zeros((4,3)))
    for step,scan in enumerate(synthetic_history(count)):
        event=PublicEvent('endurance',step,2*step+1,'shared_neighbor26','scan',scan,payload_digest(scan))
        transition=None
        if step:
            obs=StepObservation(agents,agents,velocity,velocity,False,False,True,True,False,'running')
            transition=PublicEvent('endurance',step,2*step+2,'team_search_detector','step',obs,payload_digest(obs))
        result.append(PublicStepEvidence('endurance',step,(event,),transition))
    return result


def run(output,count=640):
    from chapter3_bser.learning.provenance import verify_reviewed_changes,require_review
    if type(count) is not int or not 513<=count<=1024:raise ValueError('513..1024 public batches required')
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'work'):raise ValueError('fresh Active/work output required')
    output.mkdir(parents=True,exist_ok=False);source=verify_reviewed_changes()
    matrix=json.loads((ROOT/'configs/bounded_design_experiment_v1.json').read_text())
    stream=packets(count);ct=contract();ct.update(CONTRACT)
    evidence_sha=digest([dict(step=p.physical_step,scans=[s.payload_sha256 for s in p.scans]) for p in stream])
    rows=[];traces=[];failures=[]
    for arm,config in matrix['arms'].items():
        # V2 reference has the same public input and finite execution budgets;
        # no forced faults are injected into this timing reference.
        modes=('natural',) if arm=='reference_v2' else ('natural','forced_depletion')
        for mode in modes:
            b=(CoupledBelief if arm=='reference_v2' else BoundedCoupledBelief)(copy.deepcopy(config),ct,specs())
            timings=[];start=time.perf_counter();attempted=0;injected=[]
            for packet in stream:
                if mode=='forced_depletion' and packet.physical_step in (192,384,576):
                    b.positions[:]=[18.,18.,9.];b.velocities[:]=0.;injected.append(packet.physical_step)
                before=time.perf_counter();b.consume(packet);elapsed=time.perf_counter()-before;attempted+=1
                row=dict(arm=arm,mode=mode,step=packet.physical_step,seconds=elapsed,valid=b.valid,
                    retained_history=len(b.history),replay_work=getattr(b,'last_replay_work',0),
                    boxes=max((len(m.boxes) for m in b.maps),default=0))
                timings.append(row);traces.append(row)
                if not b.valid:break
            summary=b.summary();expected_negative=(arm=='no_local_recovery_w64' and mode=='forced_depletion')
            required=(arm in ('full_w32','full_w64','full_w128'))
            passed=(not b.valid and summary['error']=='local_recovery_disabled_ablation') if expected_negative else b.valid and attempted==count
            if (required or expected_negative) and not passed:failures.append(dict(arm=arm,mode=mode,error=summary.get('error')))
            row=dict(arm=arm,mode=mode,attempted_batches=attempted,completed_batches=attempted-int(not b.valid),
                complete=b.valid and attempted==count,expected_negative_control=expected_negative,check_passed=passed,
                injected_steps=injected,seconds=time.perf_counter()-start,**timing_bands(timings),summary=summary)
            rows.append(row)
    result=dict(passed=not failures,source=source,batches=count,rays_per_batch=104,evidence_sha256=evidence_sha,
        rows=rows,required_failures=failures,physical_episodes=0,models_created=0,
        fault_scope='internal particle depletion; not physical closed-loop or task performance',performance_passed=False)
    require_review(source);write_json(output/'endurance.json',result)
    with (output/'steps.csv').open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(traces[0]));writer.writeheader();writer.writerows(traces)
    lines=['# Public evidence endurance','',
        'Natural and injected-depletion runs are separate. Failures and censored references remain visible.',
        'Timings measure inference only; forced faults do not estimate real-world recovery frequency.','',
        '| Arm | Mode | Batches | Complete | Early p95 s | Late p95 s | Error |','|---|---|---:|---|---:|---:|---|']
    for r in rows:lines.append(f'| {r["arm"]} | {r["mode"]} | {r["attempted_batches"]} | {r["complete"]} | {r["early_p95"]:.5f} | {r["late_p95"]:.5f} | {r["summary"].get("error")} |')
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--batches',type=int,default=640)
    p.add_argument('--execute',action='store_true');args=p.parse_args(argv)
    if not args.execute:print(json.dumps(dict(executed=False,physical_episodes=0,**vars(args))));return
    result=run(args.output,args.batches)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
