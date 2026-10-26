"""Four paired ZERO controls: original/bounded map x standby guard off/on."""
import argparse
import copy
import json
from pathlib import Path
from project_paths import ROOT
from chapter3_bser.learning import map_suite as suite
from chapter3_bser.learning.checkpoint import digest, write_json
from chapter3_bser.learning.evaluation_records import verified_rows
from chapter3_bser.learning.map_report import summarize, csv_file, paired_interval
from chapter3_bser.learning.provenance import verify_reviewed_changes, require_review
from chapter3_bser.safety.executor_standby import OPTIONS


def prepare(root):
    root = Path(root).resolve()
    if root.exists():raise FileExistsError(root)
    if 'collision_terminal' not in root.parts:raise ValueError('collision_terminal root required')
    source = verify_reviewed_changes()
    spec = suite.read(ROOT/'configs/executor_standby_pilot_v1.json')
    reference = copy.deepcopy(spec)
    reference['executor_safety']['enabled'] = False
    suite.prepare(root/'guard_off', reference)
    # Freeze exactly the same physical scenarios and IDs in the second arm.
    suite.prepare(root/'guard_on', spec, evaluation_manifest=suite.read(root/'guard_off/inputs/evaluation.json'))
    plans = {name:suite.load_plan(root/name)['sha256'] for name in ('guard_off','guard_on')}
    data = dict(schema='auv.standby_experiment.v1', source_review=source, plans=plans,
        groups=['OLD_ZERO','NEW_ZERO'], paired_physical_inputs=True, training_jobs=0,
        evaluations=48, performance_passed=False)
    write_json(root/'experiment.json', data)
    return data


def load(root):
    root = Path(root).resolve();data = suite.read(root/'experiment.json')
    require_review(data['source_review'])
    if data['schema'] != 'auv.standby_experiment.v1':raise ValueError('invalid standby experiment')
    plans = {name:suite.load_plan(root/name) for name in ('guard_off','guard_on')}
    if {n:p['sha256'] for n,p in plans.items()} != data['plans']:raise ValueError('standby plans changed')
    if suite.read(root/'guard_off/inputs/evaluation.json') != suite.read(root/'guard_on/inputs/evaluation.json'):
        raise ValueError('standby physical inputs are not paired')
    for name,p in plans.items():
        spec = copy.deepcopy(p['spec'])
        if spec.pop('executor_safety') != dict(OPTIONS, enabled=name == 'guard_on'):
            raise ValueError('standby control changed')
        if name == 'guard_off':reference = spec
        elif spec != reference:raise ValueError('standby arms differ beyond guard')
    return data, plans


def episode_record(arm, group, row):
    guard = row.get('planner', {}).get('executor_standby_guard')
    last = None if guard is None else guard['last_search_guidance']
    counts = {} if guard is None else guard['counts']
    executor_collision = (row.get('first_collision_phase') == 'Search'
                          and 3 in row.get('first_collision_agent_ids', []))
    return dict(arm=arm, group=group, scenario_id=row['scenario_id'],
        length=row['actual_length'], found_step=row['found_step'], success=row['success'],
        termination_reason=row['termination_reason'], first_collision_step=row.get('first_collision_step'),
        first_collision_agents=json.dumps(row.get('first_collision_agent_ids', [])),
        first_collision_phase=row.get('first_collision_phase'), executor_search_collision=executor_collision,
        guard_available=guard is not None, guard_checks=counts.get('checks', 0),
        route_queries=counts.get('route_queries', 0), route_repairs=counts.get('route_repairs', 0),
        nominal_braking_checked=counts.get('nominal_braking_checked', 0),
        unverified_braking=counts.get('unverified_braking', 0),
        last_guard_step=None if last is None else last['step'],
        last_guard_status=None if last is None else last['status'],
        last_guard_cause=None if last is None else last['cause'])


def report(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    data, plans = load(root)
    # Native receipt and policy/condition checks precede every new table.
    output.mkdir(parents=True, exist_ok=False)
    rows = [];by_arm = {}
    for arm, plan in plans.items():
        summarize(root/arm, output/arm)
        for job in plan['jobs']:
            values, _ = verified_rows(root/arm/job['evaluation']/'episodes.jsonl', data['source_review'])
            records = [episode_record(arm, job['group'], r) for r in values]
            by_arm[(arm,job['group'])] = records
            rows.extend(records)
    comparisons = []
    for group in data['groups']:
        off, on = (by_arm[(arm,group)] for arm in ('guard_off','guard_on'))
        if [r['scenario_id'] for r in off] != [r['scenario_id'] for r in on]:raise ValueError('scene ordering changed')
        for metric in ('executor_search_collision','success'):
            delta = [[int(b[metric])-int(a[metric]) for a,b in zip(off,on)]]
            ci = paired_interval(delta, draws=2000, seed=20261026)
            ci.update(training_replicates=0, interval_scope='paired development-scene uncertainty; zero residual policy')
            comparisons.append(dict(group=group, comparison='guard_on minus guard_off', metric=metric, **ci))
    csv_file(output/'executor_collisions.csv', rows)
    csv_file(output/'guard_paired_deltas.csv', comparisons)
    result = dict(schema='auv.standby_report.v1', source_review=data['source_review'],
        complete=True, comparisons=comparisons, performance_passed=False,
        scope='Development scenarios reused to diagnose executor standby; no held-out acceptance or learned policy evidence.')
    write_json(output/'summary.json', result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','run','report'):
        sub=subs.add_parser(name);sub.add_argument('--root',required=True);sub.add_argument('--execute',action='store_true')
        if name=='run':
            sub.add_argument('--arms',nargs='+',choices=('guard_off','guard_on'),default=['guard_off','guard_on'])
            sub.add_argument('--validation',required=True)
        if name=='report':sub.add_argument('--output',required=True)
    args=vars(parser.parse_args(argv));execute=args.pop('execute');command=args.pop('command')
    if not execute:result=dict(executed=False,command=command,**args)
    elif command=='prepare':result=prepare(args['root'])
    elif command=='report':result=report(**args)
    else:
        load(args['root'])
        result=dict(executed=True,arms={arm:suite.run(Path(args['root'])/arm,stage='evaluate',validation=args['validation'])
                    for arm in args['arms']},performance_passed=False)
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':main()
