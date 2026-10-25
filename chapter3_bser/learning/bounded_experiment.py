"""Explicit bounded-map experiments; native fresh policies and paired tasks."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from project_paths import ROOT
from . import map_suite as suite
from .checkpoint import digest,write_json
from .provenance import verify_reviewed_changes,require_review
from .map_report import metric_values,paired_interval,csv_file,summarize as native_report

SCHEMA='auv.active.bounded_design.v1'


def specifications(matrix,stage):
    if matrix.get('schema')!=SCHEMA or stage not in matrix['stages']:raise ValueError('invalid bounded experiment stage')
    base=matrix['stages'][stage];result={}
    for name,config in matrix['arms'].items():
        if not name.replace('_','').isalnum():raise ValueError('unsafe experiment arm')
        spec=copy.deepcopy(base);spec['new_map_config']=config
        result[name]=suite.validate_spec(spec)
    if 'full_w64' not in result or 'reference_v2' not in result:raise ValueError('full and reference arms required')
    return result


def prepare(root,spec,stage='pilot'):
    source=verify_reviewed_changes();root=Path(root).resolve();matrix=suite.read(spec)
    configs=specifications(matrix,stage)
    if root.exists():raise FileExistsError(root)
    if 'collision_terminal' not in root.parts:raise ValueError('experiment root must contain collision_terminal')
    root.mkdir(parents=True,exist_ok=False);plans={};shared=None
    for arm,config in configs.items():
        destination=root/arm
        kwargs={} if shared is None else shared
        suite.prepare(destination,config,**kwargs)
        plan=suite.load_plan(destination);plans[arm]=plan['sha256']
        if shared is None:
            shared=dict(training_manifests={s:suite.read(destination/f'inputs/train_seed{s}.json') for s in config['seeds']},
                evaluation_manifest=suite.read(destination/'inputs/evaluation.json'),
                validation_manifest=suite.read(destination/'inputs/validation.json'))
    index=dict(schema=SCHEMA,source_review=source,stage=stage,matrix=matrix,plans=plans,
        native_policies=True,performance_passed=False)
    index['sha256']=digest(index);require_review(source);write_json(root/'design.json',index)
    load(root)
    return dict(executed=True,root=str(root),stage=stage,arms=list(plans),
        training_jobs=sum(len([j for j in suite.jobs_for(c) if j['seed'] is not None]) for c in configs.values()),
        formal_training_started=False,performance_passed=False)


def load(root):
    root=Path(root).resolve();value=suite.read(root/'design.json')
    if value.get('schema')!=SCHEMA or value.get('sha256')!=digest({k:v for k,v in value.items() if k!='sha256'}):raise ValueError('design identity mismatch')
    require_review(value['source_review']);configs=specifications(value['matrix'],value['stage'])
    if set(value['plans'])!=set(configs):raise ValueError('arm inventory changed')
    pair=None
    for arm,spec in configs.items():
        plan=suite.load_plan(root/arm)
        if plan['sha256']!=value['plans'][arm] or plan['spec']!=spec:raise ValueError('arm specification changed')
        keys=['inputs/evaluation.json','inputs/validation.json',*(f'inputs/train_seed{s}.json' for s in spec['seeds'])]
        assets={k:plan['assets'][k] for k in keys}
        if pair is not None and pair!=assets:raise ValueError('arms do not share identical physical tasks and order')
        pair=assets
    return value


def timing_bands(trace):
    out={}
    for name,part in zip(('early','middle','late'),np.array_split(np.arange(len(trace)),3)):
        values=[trace[i]['seconds'] for i in part]
        out[name+'_p50']=None if not values else float(np.percentile(values,50))
        out[name+'_p95']=None if not values else float(np.percentile(values,95))
    return out


def report(root,output):
    from .evaluation_records import verified_rows
    root=Path(root).resolve();index=load(root);output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True,exist_ok=False)
    status=[];resources=[];data={};comparisons=[]
    for arm in index['plans']:
        try:
            # Verify training, final policy, scenes, metrics and convergence;
            # incomplete arms remain visible instead of survivor-only reporting.
            native_report(root/arm,output/arm)
            plan=suite.load_plan(root/arm)
            for job in plan['jobs']:
                rows,_=verified_rows(root/arm/job['evaluation']/'episodes.jsonl',index['source_review'])
                data[(arm,job['learner'],job['seed'])]=metric_values(rows)
                for row in rows:
                    health=row['map_metrics'];b=health.get('bounded_design',{});counts=b.get('counts',{})
                    resources.append(dict(arm=arm,learner=job['learner'],training_seed=job['seed'],scenario_id=row['scenario_id'],
                        steps=row['actual_length'],window=b.get('window_steps'),max_history=b.get('max_retained_history'),
                        boxes=b.get('max_map_boxes'),snapshot_bytes=b.get('snapshot_array_bytes'),
                        cells=b.get('persistent_map',{}).get('observed_cells'),
                        recoveries=counts.get('local_recoveries',0),replay_work=counts.get('bounded_replay_steps',0),
                        transports=counts.get('approximate_geometry_transports',0),
                        **timing_bands(health.get('inference_step_trace',[]))))
            status.append(dict(arm=arm,complete=True,error=None,reported_failures=None))
        except (OSError,ValueError,RuntimeError,KeyError) as error:
            # Preserve partial/failure files in their original job directories.
            failures=[]
            for path in sorted((root/arm/'jobs').rglob('failure.json')):
                try:
                    failure=suite.read(path)
                    failures.append(dict(path=path.relative_to(root).as_posix(),
                        error=failure.get('error'),environment_step=failure.get('environment_step')))
                except (OSError,ValueError):pass
            status.append(dict(arm=arm,complete=False,error=str(error),
                reported_failures=json.dumps(failures) if failures else None))
    stage=index['matrix']['stages'][index['stage']]
    for arm in index['plans']:
        if arm=='full_w64':continue
        for learner in ('ZERO','B2','B3'):
            seeds=[None] if learner=='ZERO' else stage['seeds']
            if not all((a,learner,s) in data for a in (arm,'full_w64') for s in seeds):continue
            for metric in next(iter(data.values())):
                delta=np.stack([data[('full_w64',learner,s)][metric]-data[(arm,learner,s)][metric] for s in seeds])
                ci=paired_interval(delta,draws=stage['bootstrap_draws'],seed=stage['statistics_seed'])
                if learner=='ZERO':ci.update(training_replicates=0,interval_scope='paired scenes; no learned policy')
                comparisons.append(dict(comparison='full_w64 minus '+arm,learner=learner,metric=metric,**ci))
    result=dict(schema=SCHEMA,source_review=index['source_review'],design_sha256=index['sha256'],stage=index['stage'],
        complete=all(s['complete'] for s in status),arms=status,comparisons=comparisons,performance_passed=False)
    require_review(index['source_review']);write_json(output/'summary.json',result)
    csv_file(output/'paired_deltas.csv',comparisons);csv_file(output/'resources.csv',resources);csv_file(output/'arms.csv',status)
    lines=['# Bounded map design experiment','',
        'Incomplete arms are retained; no performance acceptance is asserted.',
        'Timing bands divide each completed episode into thirds. Compare long searches separately from early termination.',
        'no_boundary changes recovery proposals only; check that recoveries actually occurred before interpreting that ablation.',
        'Window map uses bounded preprocessed scan frames; its extra frame bytes and recomposition work belong to that ablation.','',
        '| Arm | Complete | Error |','|---|---|---|']
    lines.extend(f'| {s["arm"]} | {s["complete"]} | {s["error"] or ""} |' for s in status)
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--root',required=True);prep.add_argument('--stage',choices=('pilot','formal'),default='pilot')
    prep.add_argument('--spec',default=str(ROOT/'configs/bounded_design_experiment_v1.json'))
    run=sub.add_parser('run');run.add_argument('--root',required=True);run.add_argument('--arms',nargs='+',required=True)
    run.add_argument('--stage',choices=('train','diagnose','evaluate','all'),default='all');run.add_argument('--validation',required=True)
    summary=sub.add_parser('report');summary.add_argument('--root',required=True);summary.add_argument('--output',required=True)
    for s in (prep,run,summary):s.add_argument('--execute',action='store_true')
    args=vars(p.parse_args(argv));execute=args.pop('execute');command=args.pop('command')
    if not execute:result=dict(executed=False,command=command,**args)
    elif command=='prepare':result=prepare(**args)
    elif command=='report':result=report(**args)
    else:
        root=Path(args.pop('root')).resolve();index=load(root);arms=args.pop('arms')
        if len(arms)!=len(set(arms)) or set(arms)-set(index['plans']):raise ValueError('unique prepared arms required')
        result=dict(executed=True,arms={arm:suite.run(root/arm,**args) for arm in arms},performance_passed=False)
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':main()
