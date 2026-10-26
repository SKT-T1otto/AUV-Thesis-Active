"""Prepare/reproduce/report frozen OLD_ZERO executor collisions; no training."""
import argparse
import copy
import json
from pathlib import Path
import shutil
from project_paths import ROOT
from tools.executor_trace import read, write, sha, digest, run_episode, behavioral_summary, verified_recording

SCHEMA = 'auv.executor_diagnostics.plan.v1'
CASES = {'baseline_0002': ('guard_off','0002'), 'baseline_0006': ('guard_off','0006'),
         'guard_v1_0011': ('guard_on','0011')}
EVENT_FIELDS = ('actual_length','found_step','success','termination_reason','first_collision_step',
                'first_collision_phase','first_collision_agent_ids','collision_records')


def output_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT/'work') or 'collision_terminal' not in path.parts:
        raise ValueError('fresh Active/work/collision_terminal output required')
    if path.exists():
        raise FileExistsError(path)
    return path


def historical_inputs(root, arm):
    from chapter3_bser.learning.config import validate
    from chapter3_bser.learning.scenarios import validate_manifest
    from chapter3_bser.learning.evaluation_records import verified_rows
    from chapter3_bser.learning.map_contract import identity as condition_identity
    from chapter3_bser.learning.checkpoint import digest as native_digest
    parent = ROOT/'provenance/executor_standby_manifest.json'
    plan = read(root/'plan.json')
    if plan.get('schema') != 'auv.active.map_suite.v1' or plan.get('sha256') != native_digest({k:v for k,v in plan.items() if k!='sha256'}):
        raise ValueError('invalid historical native plan')
    if plan['source_review']['review_sha256'] != read(parent)['sha256']:
        raise ValueError('historical source is not the verified aef6833 standby experiment')
    job = next(j for j in plan['jobs'] if j['name']=='OLD_ZERO')
    if job['config'] != 'inputs/OLD_ZERO.json' or job['evaluation'] != 'jobs/OLD_ZERO/evaluation':
        raise ValueError('unsafe historical OLD_ZERO paths')
    for name in ('inputs/evaluation.json', job['config']):
        if sha(root/name) != plan['assets'][name]:
            raise ValueError('frozen historical input changed: '+name)
    config = validate(read(root/job['config']))
    condition = config['map_experiment']
    if condition['map_id']!='OLD' or condition['research_config'] is not None or config['max_steps']!=400:
        raise ValueError('only original OLD 400-step condition allowed')
    from chapter3_bser.safety.executor_standby import OPTIONS
    if condition.get('executor_safety') != dict(OPTIONS,enabled=arm=='guard_on'):
        raise ValueError('original/v1 guard condition changed; v2 is excluded')
    if condition_identity(config)!=job['map_condition']:
        raise ValueError('historical native map condition mismatch')
    manifest = read(root/'inputs/evaluation.json')
    scenes = validate_manifest(manifest,config,evaluation=True)
    rows, receipt = verified_rows(root/job['evaluation']/'episodes.jsonl',plan['source_review'])
    if receipt['method']!='D2' or receipt['manifest_sha256']!=native_digest(manifest) or len(rows)!=len(scenes):
        raise ValueError('historical native ZERO evaluation binding changed')
    if receipt.get('map_condition')!=job['map_condition'] or any(r.get('map_condition')!=job['map_condition'] for r in rows):
        raise ValueError('historical evaluation condition changed')
    if any(r['scenario_id']!=s['scenario_id'] or r['scenario_sha256']!=native_digest(s) for r,s in zip(rows,scenes)):
        raise ValueError('historical scenario order/bytes changed')
    return config, scenes, rows


def prepare(experiment, root, *, cases=None):
    from chapter3_bser.learning.provenance import verify_reviewed_changes, require_review
    source = verify_reviewed_changes()
    root, experiment = output_path(root), Path(experiment).resolve()
    if not experiment.is_relative_to(ROOT/'work'):
        raise ValueError('native historical experiment must be inside this Active/work')
    chosen = list(CASES if cases is None else cases)
    if not chosen or len(set(chosen))!=len(chosen) or set(chosen)-CASES.keys():
        raise ValueError('unique supported diagnosis cases required')
    inputs, material = root/'inputs', {}
    # Verify original evidence before creating any diagnosis output.
    for arm in sorted({CASES[c][0] for c in chosen}):
        material[arm] = historical_inputs(experiment/arm,arm)
    inputs.mkdir(parents=True,exist_ok=False)
    for arm in material:
        for name in ('plan.json','inputs/evaluation.json','inputs/OLD_ZERO.json',
                     'jobs/OLD_ZERO/evaluation/evaluation_identity.json','jobs/OLD_ZERO/evaluation/episodes.jsonl'):
            destination=inputs/'history'/arm/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(experiment/arm/name,destination)
    records=[]
    for name in chosen:
        arm,suffix=CASES[name]
        config,scenes,rows=material[arm]
        scene=next(s for s in scenes if s['scenario_id'].endswith('_'+suffix))
        expected=next(r for r in rows if r['scenario_id']==scene['scenario_id'])
        c=copy.deepcopy(config);c['output_dir']=str(root/'jobs'/name/'unused_train')
        write(inputs/(name+'_config.json'),c)
        write(inputs/(name+'_scene.json'),scene)
        write(inputs/(name+'_expected.json'),expected)
        records.append(dict(name=name,arm=arm,scenario_id=scene['scenario_id'],
            config='inputs/'+name+'_config.json',scene='inputs/'+name+'_scene.json',
            expected='inputs/'+name+'_expected.json'))
    plan=dict(schema=SCHEMA,source=source,cases=records,paired_recording=True,
        historical_commit='aef6833a6b80dcede92561b6497aafc1d5ad60e7',
        max_episode_steps=400,training_jobs=0,checkpoint_restore=False,
        assets={p.relative_to(root).as_posix():sha(p) for p in inputs.rglob('*') if p.is_file()})
    plan['sha256']=digest(plan);require_review(source);write(root/'plan.json',plan)
    load_plan(root)
    return dict(executed=True,root=str(root),cases=chosen,episodes=2*len(chosen),
                max_episode_steps=400,training_jobs=0,performance_passed=False)


def load_plan(root):
    from chapter3_bser.learning.provenance import require_review
    root=Path(root).resolve();plan=read(root/'plan.json')
    if plan.get('schema')!=SCHEMA or plan.get('sha256')!=digest({k:v for k,v in plan.items() if k!='sha256'}):
        raise ValueError('diagnosis plan digest changed')
    require_review(plan['source'])
    assets={p.relative_to(root).as_posix():sha(p) for p in (root/'inputs').rglob('*') if p.is_file()}
    if assets!=plan['assets'] or plan['paired_recording'] is not True or plan['training_jobs']!=0:
        raise ValueError('diagnosis input inventory changed')
    if len({c['name'] for c in plan['cases']})!=len(plan['cases']):
        raise ValueError('repeated diagnosis case')
    historical={}
    for case in plan['cases']:
        if case['name'] not in CASES or case['arm']!=CASES[case['name']][0]:
            raise ValueError('unsupported diagnosis case')
        for field in ('config','scene','expected'):
            if case[field]!='inputs/'+case['name']+'_'+field+'.json':
                raise ValueError('unsafe diagnosis case path')
        arm=case['arm']
        if arm not in historical:
            historical[arm]=historical_inputs(root/'inputs/history'/arm,arm)
        config,scenes,rows=historical[arm]
        actual=read(root/case['config']);actual.pop('output_dir')
        original=copy.deepcopy(config);original.pop('output_dir')
        if actual!=original or read(root/case['config'])['output_dir']!=str(root/'jobs'/case['name']/'unused_train'):
            raise ValueError('runtime config differs beyond output metadata')
        scene=read(root/case['scene']);expected=read(root/case['expected'])
        if scene!=next(s for s in scenes if s['scenario_id']==case['scenario_id']) or expected!=next(r for r in rows if r['scenario_id']==case['scenario_id']):
            raise ValueError('case scene or original expected event changed')
    return plan


def event_comparison(actual,expected):
    if any(k not in actual or k not in expected for k in EVENT_FIELDS):
        raise ValueError('incomplete native terminal event fields')
    checks={k:actual.get(k)==expected.get(k) for k in EVENT_FIELDS}
    return dict(exact_terminal_event_match=all(checks.values()),fields=checks,
        expected={k:expected.get(k) for k in EVENT_FIELDS},actual={k:actual.get(k) for k in EVENT_FIELDS},
        historical_full_trajectory_equality_established=False)


def run(root, validation):
    from chapter3_bser.learning.provenance import require_review, research_readiness
    from chapter3_bser.learning.map_suite import require_linux_validation
    from chapter3_bser.learning.suite import run_lock
    root=Path(root).resolve();plan=load_plan(root)
    require_linux_validation(validation,plan['source']);research_readiness(plan['source'])
    if (root/'run.json').exists():
        raise FileExistsError(root/'run.json')
    records=[]
    with run_lock(root):
        for case in plan['cases']:
            load_plan(root)
            config,scene,expected=(read(root/case[k]) for k in ('config','scene','expected'))
            summaries,signatures=[],[]
            receipts={}
            for mode in ('plain','recorded'):
                print('[executor diagnosis] '+case['name']+' '+mode,flush=True)
                directory=root/'jobs'/case['name']/mode
                summary,steps=run_episode(config,scene,directory,instrumented=mode=='recorded')
                summaries.append(summary);signatures.append(steps)
                receipts[mode]=sha(directory/'recording.json')
            comparisons=[event_comparison(s,expected) for s in summaries]
            records.append(dict(name=case['name'],scenario_id=case['scenario_id'],
                exact_step_signature_parity=signatures[0]==signatures[1],
                behavioral_summary_parity=behavioral_summary(summaries[0])==behavioral_summary(summaries[1]),
                ignored_wall_timer_fields=['compile_seconds','max_compile_seconds'],
                original_event_comparisons=comparisons,receipts=receipts))
        require_review(plan['source']);require_linux_validation(validation,plan['source'])
        result=dict(schema='auv.executor_diagnostics.run.v1',completed=True,source=plan['source'],plan_sha256=plan['sha256'],
            validation_sha256=sha(validation),cases=records,episodes=2*len(records),training_jobs=0,
            recording_parity_passed=all(r['exact_step_signature_parity'] and r['behavioral_summary_parity'] for r in records),
            all_original_terminal_events_exact=all(c['exact_terminal_event_match'] for r in records for c in r['original_event_comparisons']),
            causal_ranking_established=False,performance_passed=False)
        write(root/'run.json',result)
    return result


def report(root, output):
    from chapter3_bser.learning.provenance import require_review
    from tools.executor_trace_analysis import analyze_episode
    root,output=Path(root).resolve(),output_path(output)
    plan=load_plan(root);result=read(root/'run.json')
    if not result.get('completed') or result.get('source')!=plan['source'] or result.get('plan_sha256')!=plan['sha256']:
        raise ValueError('incomplete or changed diagnostic run')
    if [r['name'] for r in result['cases']]!=[c['name'] for c in plan['cases']]:
        raise ValueError('diagnostic case inventory changed')
    for item in result['cases']:
        for mode in ('plain','recorded'):
            path=root/'jobs'/item['name']/mode
            if sha(path/'recording.json')!=item['receipts'][mode]:
                raise ValueError('diagnostic receipt changed')
            verified_recording(path)
        case=next(c for c in plan['cases'] if c['name']==item['name'])
        summaries=[read(root/'jobs'/item['name']/mode/'episode.json') for mode in ('plain','recorded')]
        signatures=[read(root/'jobs'/item['name']/mode/'signatures.json') for mode in ('plain','recorded')]
        if (item['exact_step_signature_parity']!=(signatures[0]==signatures[1])
            or item['behavioral_summary_parity']!=(behavioral_summary(summaries[0])==behavioral_summary(summaries[1]))
            or item['original_event_comparisons']!=[event_comparison(s,read(root/case['expected'])) for s in summaries]):
            raise ValueError('diagnostic comparison claims changed')
    if (result['recording_parity_passed']!=all(r['exact_step_signature_parity'] and r['behavioral_summary_parity'] for r in result['cases'])
        or result['all_original_terminal_events_exact']!=all(c['exact_terminal_event_match'] for r in result['cases'] for c in r['original_event_comparisons'])):
        raise ValueError('diagnostic aggregate claims changed')
    output.mkdir(parents=True,exist_ok=False)
    analyses=[]
    for case in plan['cases']:
        analysis=analyze_episode(root/'jobs'/case['name']/'recorded',read(root/case['scene']),output/case['name'])
        analyses.append(dict(name=case['name'],**analysis))
    require_review(plan['source'])
    summary=dict(schema='auv.executor_diagnostics.report.v1',completed=True,run_sha256=sha(root/'run.json'),
        recording_parity_passed=result['recording_parity_passed'],
        all_original_terminal_events_exact=result['all_original_terminal_events_exact'],
        analyses=analyses,causal_ranking_established=False,performance_passed=False,
        interpretation='Review target/path/map/motion evidence jointly; mismatched reproduction or observer parity cannot establish the historical cause.')
    write(output/'summary.json',summary)
    write(output/'report_receipt.json',dict(source=plan['source'],files={p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()}))
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
    for name in ('prepare','run','report'):
        sub=subs.add_parser(name);sub.add_argument('--root',required=True);sub.add_argument('--execute',action='store_true')
        if name=='prepare':
            sub.add_argument('--experiment',required=True);sub.add_argument('--cases',nargs='+',choices=tuple(CASES))
        if name=='run':sub.add_argument('--validation',required=True)
        if name=='report':sub.add_argument('--output',required=True)
    args=vars(p.parse_args(argv));execute=args.pop('execute');command=args.pop('command')
    if not execute:result=dict(executed=False,command=command,**args)
    else:result={'prepare':prepare,'run':run,'report':report}[command](**args)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if execute and command=='run' and not result['recording_parity_passed']:
        raise SystemExit(1)
    return result


if __name__=='__main__':main()
