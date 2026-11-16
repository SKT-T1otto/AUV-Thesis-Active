"""Explicit same-scene rule comparisons. This tool cannot start training."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from project_paths import ROOT
from tools.path_option_experiment import read,generate
from chapter3_bser.r123.source import identity as parent_identity
from chapter3_bser.r123.options import CONTROL,options
from chapter3_bser.learning.config import scenario_fingerprint
from chapter3_bser.path_option.workflow import task_config,check_splits
from chapter3_bser.r123_repair.runtime import VARIANTS
from chapter3_bser.r123_repair.source import identity,admission,sha,digest,write,new_output
from chapter3_bser.r123_repair.workflow import evaluate,metric


def prepare(args):
    source=identity();out=new_output(args.output)
    config=task_config(read(args.config));opts=options(read(args.options))
    manifest=read(args.development) if args.development else generate(6,992001,'eval')
    scenes=manifest['scenarios']
    if len(scenes)!=6:raise ValueError('exactly six development scenes required')
    check_splits([],scenes,[])
    baseline=None
    if args.legacy_reference:
        base=Path(args.legacy_reference);evaluation=read(base/'evaluation.json')
        if (evaluation['source']!=parent_identity() or evaluation['config_hash']!=digest(config) or
            evaluation['options']!=opts or evaluation['scenario_fingerprints']!=[scenario_fingerprint(s) for s in scenes] or
            evaluation['seed'] is not None or evaluation['arm']!='rule' or
            evaluation['summary_sha256']!=sha(base/'summary.json')):
            raise ValueError('historical rule reference settings/source/scenes mismatch')
        records=[json.loads(s) for s in (base/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
        if [r['scenario_id'] for r in records]!=[s['scenario_id'] for s in scenes] or any(r['actual_length']>config['max_steps'] for r in records):
            raise ValueError('historical episodes do not match requested scenes')
        baseline=dict(source=parent_identity(),evaluation_sha256=sha(base/'evaluation.json'),
            episodes_sha256=sha(base/'episodes.jsonl'),metrics=[metric(r,config['max_steps']) for r in records],
            scope='retained historical rule; not rerun under new source')
    out.mkdir(parents=True)
    write(out/'config.json',config);write(out/'options.json',opts);write(out/'development.json',manifest)
    if baseline is not None:write(out/'historical_reference.json',baseline)
    value=dict(source=source,control=CONTROL,variants=list(VARIANTS),episodes_per_variant=6,
        maximum_physical_steps_per_variant=6*config['max_steps'],training_allowed=False,
        inputs={p.name:sha(p) for p in out.glob('*.json')})
    write(out/'plan.json',value);return value


def frozen(path):
    root=Path(path);plan=read(root/'plan.json')
    if plan['source']!=identity() or plan['control']!=CONTROL:raise ValueError('new repair source plan required')
    for name,h in plan['inputs'].items():
        if sha(root/name)!=h:raise ValueError('frozen comparison input changed: '+name)
    return root,plan


def evaluation(args):
    admission(args.validation);root,plan=frozen(args.root)
    return evaluate(read(root/'config.json'),read(root/'development.json')['scenarios'],
        read(root/'options.json'),args.variant,new_output(args.output))


def report(args):
    root,plan=frozen(args.root);out=new_output(args.output);out.mkdir(parents=True)
    groups={};sources={}
    if (root/'historical_reference.json').is_file():
        old=read(root/'historical_reference.json');groups['legacy']=old['metrics'];sources['legacy']=old['source']
    for path in args.results:
        p=Path(path);r=read(p/'completed.json')
        if (r['source']!=identity() or r['config_hash']!=digest(read(root/'config.json')) or
            r['options_hash']!=digest(read(root/'options.json')) or r['seed']!=982001 or
            r['scenario_ids']!=[s['scenario_id'] for s in read(root/'development.json')['scenarios']]):raise ValueError('comparison settings mismatch')
        for name,key in [('summary.json','summary_sha256'),('episodes.jsonl','episodes_sha256'),('metrics.jsonl','metrics_sha256')]:
            if sha(p/name)!=r[key]:raise ValueError('result artifact changed')
        if r['variant'] in groups:raise ValueError('duplicate variant result; do not replace retained reference silently')
        groups[r['variant']]=[json.loads(s) for s in (p/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        sources[r['variant']]=r['source']
    contrasts={}
    for left,right in [('repaired','legacy'),('no_recovery','legacy'),('repaired','no_recovery')]:
        if left not in groups or right not in groups:continue
        if [r['scenario_id'] for r in groups[left]]!=[r['scenario_id'] for r in groups[right]]:raise ValueError('unpaired scenes')
        metrics={}
        for key in ('success','collision','found','timeout','search_wait_steps','recovery_returns','active_recovery_enters','connector_steps','motion_distance'):
            delta=[a[key]-b[key] for a,b in zip(groups[left],groups[right])]
            metrics[key]=dict(per_scene=delta,mean=float(np.mean(delta)))
        contrasts[left+' minus '+right]=metrics
    value=dict(source=identity(),sources=sources,results=groups,paired_contrasts=contrasts,
        interpretation='repaired minus no_recovery isolates active endpoint recovery conditional on common fixes; legacy contrasts bundle fixes',
        training_allowed=False,performance_accepted=False,missing_variants=sorted(set(VARIANTS)-groups.keys()))
    write(out/'report.json',value);return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--output',required=True);q.add_argument('--development');q.add_argument('--legacy-reference')
    q.add_argument('--config',default='configs/b2.json');q.add_argument('--options',default='configs/r123_learning_v1.json')
    q=sub.add_parser('evaluate');q.add_argument('--root',required=True);q.add_argument('--validation',required=True)
    q.add_argument('--variant',choices=VARIANTS,required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('report');q.add_argument('--root',required=True);q.add_argument('--results',nargs='*',default=[]);q.add_argument('--output',required=True)
    for q in sub.choices.values():q.add_argument('--execute',action='store_true')
    a=p.parse_args(argv)
    if not a.execute:value=dict(executed=False,requested=vars(a))
    else:
        torch.set_num_threads(1);value=dict(prepare=prepare,evaluate=evaluation,report=report)[a.command](a)
    print(json.dumps(value,indent=2,allow_nan=False));return value


if __name__=='__main__':main()
