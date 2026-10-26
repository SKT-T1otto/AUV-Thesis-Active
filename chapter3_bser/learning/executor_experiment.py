"""OLD-only zero-residual baseline versus executor recovery v2."""
import argparse
import copy
import json
from pathlib import Path
from project_paths import ROOT
from chapter3_bser.learning import map_suite as suite
from chapter3_bser.learning.checkpoint import write_json
from chapter3_bser.learning.evaluation_records import verified_rows
from chapter3_bser.learning.map_report import summarize,csv_file,paired_interval
from chapter3_bser.learning.provenance import verify_reviewed_changes,require_review
from chapter3_bser.safety.executor_recovery_options import OPTIONS

ARMS=('baseline','recovery')


def prepare(root):
    root=Path(root).resolve()
    if root.exists():raise FileExistsError(root)
    if 'collision_terminal' not in root.parts:raise ValueError('collision_terminal root required')
    source=verify_reviewed_changes();spec=suite.read(ROOT/'configs/executor_recovery_pilot_v2.json')
    off=copy.deepcopy(spec);off['executor_safety']['enabled']=False
    suite.prepare(root/'baseline',off)
    suite.prepare(root/'recovery',spec,evaluation_manifest=suite.read(root/'baseline/inputs/evaluation.json'))
    data=dict(schema='auv.executor_recovery_experiment.v2',source_review=source,
        plans={a:suite.load_plan(root/a)['sha256'] for a in ARMS},groups=['OLD_ZERO'],
        paired_physical_inputs=True,training_jobs=0,evaluations=24,performance_passed=False)
    write_json(root/'experiment.json',data);return data


def load(root):
    root=Path(root).resolve();data=suite.read(root/'experiment.json');require_review(data['source_review'])
    if data['schema']!='auv.executor_recovery_experiment.v2':raise ValueError('invalid executor experiment')
    plans={a:suite.load_plan(root/a) for a in ARMS}
    if {a:p['sha256'] for a,p in plans.items()}!=data['plans']:raise ValueError('executor plans changed')
    if suite.read(root/'baseline/inputs/evaluation.json')!=suite.read(root/'recovery/inputs/evaluation.json'):
        raise ValueError('executor physical inputs are not paired')
    for arm,plan in plans.items():
        spec=copy.deepcopy(plan['spec']);options=spec.pop('executor_safety')
        if spec['groups']!=['OLD_ZERO'] or options!=dict(OPTIONS,enabled=arm=='recovery'):
            raise ValueError('OLD-only executor condition changed')
        if arm=='baseline':reference=spec
        elif spec!=reference:raise ValueError('arms differ beyond executor guard')
    return data,plans


def record(arm,row):
    guard=row.get('planner',{}).get('executor_recovery');counts={} if guard is None else guard['counts']
    agents=row.get('first_collision_agent_ids',[]);collision=row['termination_reason']=='obstacle_collision'
    return dict(arm=arm,scenario_id=row['scenario_id'],length=row['actual_length'],found_step=row['found_step'],
        found=row['found_step'] is not None,success=row['success'],collision=collision,
        executor_collision=collision and 3 in agents,
        executor_search_collision=collision and 3 in agents and row.get('first_collision_phase')=='Search',
        first_collision_step=row.get('first_collision_step'),first_collision_phase=row.get('first_collision_phase'),
        first_collision_agents=json.dumps(agents),termination_reason=row['termination_reason'],
        guard_available=guard is not None,search_checks=counts.get('checks',0),
        move=counts.get('mode:MOVE',0),hold=counts.get('mode:HOLD',0),recover=counts.get('mode:RECOVER',0),
        unverified=counts.get('mode:UNVERIFIED',0),route_queries=counts.get('route_queries',0),
        route_repairs=counts.get('route_repairs',0),public_error_observations=counts.get('public_error_observations',0),
        compile_seconds=None if guard is None else guard['compile_seconds'],
        max_compile_seconds=None if guard is None else guard['max_compile_seconds'])


def report(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    data,plans=load(root);output.mkdir(parents=True,exist_ok=False);records={}
    for arm,plan in plans.items():
        summarize(root/arm,output/arm)
        job=plan['jobs'][0]
        rows,_=verified_rows(root/arm/job['evaluation']/'episodes.jsonl',data['source_review'])
        records[arm]=[record(arm,r) for r in rows]
    off,on=records['baseline'],records['recovery']
    if [r['scenario_id'] for r in off]!=[r['scenario_id'] for r in on]:raise ValueError('scene pairing changed')
    comparisons=[]
    for metric in ('executor_search_collision','executor_collision','collision','found','success'):
        ci=paired_interval([[int(b[metric])-int(a[metric]) for a,b in zip(off,on)]],draws=2000,seed=20261027)
        ci.update(training_replicates=0,interval_scope='paired reused development scenes; ZERO policy')
        comparisons.append(dict(metric=metric,comparison='recovery minus baseline',**ci))
    csv_file(output/'executor_episodes.csv',off+on);csv_file(output/'paired_deltas.csv',comparisons)
    result=dict(schema='auv.executor_recovery_report.v2',source_review=data['source_review'],complete=True,
        comparisons=comparisons,performance_passed=False,
        scope='OLD-only development pilot. Search and ALL-phase executor collisions are both reported; no learned-policy or formal safety acceptance.')
    write_json(output/'summary.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
    for name in ('prepare','run','report'):
        sub=subs.add_parser(name);sub.add_argument('--root',required=True);sub.add_argument('--execute',action='store_true')
        if name=='run':
            sub.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS));sub.add_argument('--validation',required=True)
        if name=='report':sub.add_argument('--output',required=True)
    args=vars(p.parse_args(argv));execute=args.pop('execute');command=args.pop('command')
    if not execute:result=dict(executed=False,command=command,**args)
    elif command=='prepare':result=prepare(args['root'])
    elif command=='report':result=report(**args)
    else:
        load(args['root'])
        result=dict(executed=True,arms={a:suite.run(Path(args['root'])/a,stage='evaluate',validation=args['validation']) for a in args['arms']},performance_passed=False)
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':main()
