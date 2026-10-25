"""Same-input inference/fault matrix. No simulator, learner or checkpoint."""
import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from project_paths import ROOT
from chapter3_bser.experiments.coupled_search_v1.bounded_belief import BoundedCoupledBelief
from chapter3_bser.learning.checkpoint import write_json
from checks.test_coupled_search import contract,specs
from tools.check_bounded_map import packets
from tools.check_map_cache import CONTRACT


def cases(matrix):
    rows=[(arm,mode) for arm in ('full_w32','full_w64','full_w128')
          for mode in ('natural','target_depletion','geometry_depletion')]
    rows += [('window_map_w64','natural'),('no_boundary_w64','target_depletion'),
             ('no_local_recovery_w64','target_depletion')]
    return [(arm,mode,copy.deepcopy(matrix['arms'][arm])) for arm,mode in rows]


def run(output,batches=384):
    from chapter3_bser.learning.provenance import verify_reviewed_changes,require_review
    if type(batches) is not int or not 257<=batches<=1024:
        raise ValueError('257..1024 public batches required')
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT/'work'):raise ValueError('fresh Active/work output required')
    source=verify_reviewed_changes();output.mkdir(parents=True,exist_ok=False)
    matrix=json.loads((ROOT/'configs/bounded_design_experiment_v1.json').read_text(encoding='utf-8'))
    stream=packets(batches);ct=contract();ct.update(CONTRACT)
    stream_path=output/'public_stream.jsonl'
    with stream_path.open('x',encoding='utf-8') as f:
        for packet in stream:f.write(json.dumps(asdict(packet),sort_keys=True)+'\n')
    stream_hash=hashlib.sha256(stream_path.read_bytes()).hexdigest()
    fault_steps=(192,min(320,batches-1));results=[];trace=[]
    for arm,mode,config in cases(matrix):
        belief=BoundedCoupledBelief(config,ct,specs());injected=[];attempted=0
        for packet in stream:
            step=packet.physical_step
            if mode!='natural' and step in fault_steps:
                if mode=='geometry_depletion':
                    boxes=belief.memory.boxes()
                    if not boxes:raise ValueError('geometry fault fixture has no occupied support')
                    belief.positions[:]=boxes[0][0]
                    belief.geometry.previous_base=None  # Explicit internal fault, never a runtime policy.
                else:belief.positions[:]=[18.,18.,9.]
                belief.velocities[:]=0.;injected.append(step)
            belief.consume(packet);attempted+=1
            event=dict(belief.inference_diagnostics.trace[-1])
            event.update(arm=arm,mode=mode,valid=belief.valid)
            trace.append(event)
            if not belief.valid:break
        summary=belief.summary();negative=arm=='no_local_recovery_w64'
        if negative:
            passed=(not belief.valid and belief.error=='local_recovery_disabled_ablation' and injected==[fault_steps[0]])
        else:
            passed=belief.valid and attempted==batches
            if mode!='natural':passed=passed and summary['counts'].get('local_recoveries',0)>=len(fault_steps)
        passed=passed and summary['max_retained_history']<=config['history_steps']
        passed=passed and summary['max_replay_steps']<=config['history_steps']
        passed=passed and max((r.get('replay_work') or 0 for r in trace if r['arm']==arm and r['mode']==mode),default=0)<=2*config['history_steps']
        results.append(dict(arm=arm,mode=mode,passed=bool(passed),expected_negative_control=negative,
            complete=belief.valid and attempted==batches,attempted_batches=attempted,
            injected_steps=injected,public_stream_sha256=stream_hash,summary=summary,
            inference=belief.inference_diagnostics.summary()))
    require_review(source)
    result=dict(schema='auv.active.map_diagnostic_probe.v1',passed=all(r['passed'] for r in results),
        source=source,batches=batches,public_stream_sha256=stream_hash,cases=results,
        physical_episodes=0,models_created=0,checkpoint_files_restored=0,performance_passed=False,
        scope='Synthetic public rays and detector trajectories are independently specified. Internal faults test mechanisms, not physical consistency, natural recovery frequency or task performance.')
    write_json(output/'diagnostic_probe.json',result)
    with (output/'attempts.csv').open('x',encoding='utf-8',newline='') as f:
        keys=('arm','mode','step','valid','completed','seconds','error','failure_phase','first_exception','phases','retained_history','replay_work')
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for row in trace:
            writer.writerow({k:json.dumps(row[k]) if isinstance(row[k],dict) else row[k] for k in keys})
    lines=['# Same-input inference diagnostic matrix','',result['scope'],'',
        '| Arm | Mode | Attempted | Complete | Expected negative | Check | Error |',
        '|---|---|---:|---|---|---|---|']
    for r in results:
        lines.append(f'| {r["arm"]} | {r["mode"]} | {r["attempted_batches"]} | {r["complete"]} | {r["expected_negative_control"]} | {r["passed"]} | {r["summary"]["error"] or ""} |')
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--batches',type=int,default=384);p.add_argument('--execute',action='store_true')
    args=p.parse_args(argv)
    if not args.execute:
        result=dict(executed=False,physical_episodes=0,models_created=0,**vars(args))
    else:
        result=run(args.output,args.batches)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))
    if args.execute and not result['passed']:raise SystemExit(1)
    return result


if __name__=='__main__':main()
