"""One explicit R123 job per command; preparation never starts training."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from project_paths import ROOT
from tools.path_option_experiment import generate, read
from chapter3_bser.r123.options import SCHEMA, ARMS, CONTROL, options, switches
from chapter3_bser.r123.source import identity, admission, new_output, sha, digest, write
from chapter3_bser.r123.workflow import train, evaluate, load_policy
from chapter3_bser.path_option.workflow import task_config, check_splits
from chapter3_bser.learning.config import scenario_fingerprint


def prepare(args):
    source=identity();out=new_output(args.output)
    config=task_config(read(args.config));opts=options(read(args.options))
    design=read(ROOT/'configs/r123_ablation_v1.json')
    if design['arms']!={arm:switches(arm) for arm in ARMS} or design['control']!=CONTROL:
        raise ValueError('ablation design differs from runtime')
    sets={}
    counts=dict(training=32,development=6,test=24)
    for name,seed in [('training',991001),('development',992001),('test',993001)]:
        print(f'[r123 prepare] {name}: {counts[name]} scenes',flush=True)
        sets[name]=generate(counts[name],seed,'train' if name=='training' else 'eval')
    check_splits(*(sets[name]['scenarios'] for name in ('training','development','test')))
    out.mkdir(parents=True)
    for name,value in sets.items():write(out/(name+'.json'),value)
    write(out/'config.json',config);write(out/'options.json',opts)
    plan=dict(schema=SCHEMA,source=source,stage=args.stage,control=CONTROL,
        arms=['full'] if args.stage=='screen' else list(ARMS),
        seeds=[20261115] if args.stage=='screen' else [20261115,20261116,20261117],
        train_step_budget=8000 if args.stage=='screen' else 20000,
        max_training_episodes=20 if args.stage=='screen' else None,
        evaluations_per_job=2,development_episodes_per_evaluation=6,
        primary_policy='fixed final training budget; no best-checkpoint selection',
        training_resume=False,test_locked_until_settings_frozen=True,
        requires_reference_gate=True,requires_screen_gate=args.stage=='attribution',
        inputs={p.name:sha(p) for p in out.glob('*.json')})
    write(out/'plan.json',plan)
    return plan


def frozen(path):
    root=Path(path).resolve();plan=read(root/'plan.json')
    if plan['source']!=identity() or plan['control']!=CONTROL:raise ValueError('plan source changed')
    for n,h in plan['inputs'].items():
        if sha(root/n)!=h:raise ValueError('frozen input changed: '+n)
    return root,plan


def evaluation(args):
    admission(args.validation);root,plan=frozen(args.root)
    if args.suite=='test' and not args.final_test:
        raise ValueError('test requires explicit --final-test after settings freeze')
    config=read(root/'config.json');scenes=read(root/(args.suite+'.json'))['scenarios']
    model=None
    if args.mode=='learned':
        if not args.policy:raise ValueError('new-source policy required')
        model=load_policy(args.policy,config,scenes)
    elif args.policy:raise ValueError('rule cannot load weights')
    arm='rule' if model is None else model.arm
    out=new_output(args.output)
    result=evaluate(config,scenes,read(root/'options.json'),model,out,arm=arm)
    if args.mode=='rule' and args.suite=='development':
        write(out/'reference_gate.json',dict(source=identity(),control=CONTROL,options_hash=digest(read(root/'options.json')),
            config_hash=digest(config),scene_ids=[s['scenario_id'] for s in scenes],
            summary_hash=sha(out/'summary.json'),evaluation_hash=sha(out/'evaluation.json'),
            passed=bool(result['success']>=.5 and result['collision']==0),
            requirement='at least3 of6 full tasks successful with no collisions; development gate only'))
    return result


def reference_gate(path,root):
    r=read(path);folder=Path(path).parent
    expected=[s['scenario_id'] for s in read(root/'development.json')['scenarios']]
    if (r.get('source')!=identity() or not r.get('passed') or r['scene_ids']!=expected or
        r['config_hash']!=digest(read(root/'config.json')) or r['options_hash']!=digest(read(root/'options.json')) or
        r['summary_hash']!=sha(folder/'summary.json') or r['evaluation_hash']!=sha(folder/'evaluation.json')):
        raise ValueError('valid matching rule full-mission reference gate required; do not train around a broken flow')


def run(args):
    admission(args.validation);root,plan=frozen(args.root);reference_gate(args.reference,root)
    if args.arm not in plan['arms'] or args.seed not in plan['seeds']:raise ValueError('arm/seed outside stage')
    if plan['stage']=='attribution':
        if not args.screen:raise ValueError('completed improved Full screen required')
        screen=read(args.screen)
        if (screen['source']!=identity() or screen['arm']!='full' or not screen['screen_passed'] or
            screen['step_budget']!=8000 or screen['config_hash']!=digest(read(root/'config.json')) or
            screen['options_hash']!=digest({k:v for k,v in read(root/'options.json').items() if k!='seed'}) or
            screen['train_fingerprints']!=[scenario_fingerprint(s) for s in read(root/'training.json')['scenarios']] or
            screen['dev_fingerprints']!=[scenario_fingerprint(s) for s in read(root/'development.json')['scenarios']]):
            raise ValueError('Full screen has not demonstrated task gain without collision regression')
    out=root/'jobs'/f'{args.arm}_seed{args.seed}';out.parent.mkdir(exist_ok=True)
    lock=out.with_suffix('.lock');fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.close(fd)
    try:
        return train(read(root/'config.json'),read(root/'training.json')['scenarios'],
            read(root/'development.json')['scenarios'],read(root/'options.json'),args.arm,out,
            seed=args.seed,step_budget=plan['train_step_budget'],max_episodes=plan['max_training_episodes'])
    except Exception:
        import traceback
        if out.exists() and not (out/'failure.json').exists():
            write(out/'failure.json',dict(traceback=traceback.format_exc(),resume=False))
        raise
    finally:lock.unlink()


def report(args):
    root,plan=frozen(args.root);out=new_output(args.output);out.mkdir(parents=True)
    groups={};jobs=[]
    for arm in plan['arms']:
        for seed in plan['seeds']:
            p=root/'jobs'/f'{arm}_seed{seed}'/'completed.json'
            if not p.exists():jobs.append(dict(arm=arm,seed=seed,completed=False));continue
            r=read(p)
            if r['source']!=identity() or r['arm']!=arm or r['training_seed']!=seed:raise ValueError('job identity mismatch')
            jobs.append(dict(arm=arm,seed=seed,completed=True,physical_steps=r['physical_steps'],initial=r['initial'],final=r['final']))
            groups.setdefault(arm,{})[seed]=r['final']
    contrasts={}
    for removed in ('no_r1','no_r2','no_r3'):
        if set(groups.get('full',{}))==set(plan['seeds'])==set(groups.get(removed,{})):
            deltas=[groups['full'][s]['success']-groups[removed][s]['success'] for s in plan['seeds']]
            metrics={}
            for key in groups['full'][plan['seeds'][0]]:
                values=[groups['full'][s][key]-groups[removed][s][key] for s in plan['seeds']]
                metrics[key]=dict(per_seed=values,mean=float(np.mean(values)),
                    seed_sd=float(np.std(values,ddof=1)) if len(values)>1 else None)
            contrasts[removed]=dict(per_seed_success_delta=deltas,mean=float(np.mean(deltas)),metrics=metrics,
                seed_sd=float(np.std(deltas,ddof=1)) if len(deltas)>1 else None,
                interpretation='conditional effect given the other two changes; not standalone factorial effect')
    value=dict(source=identity(),jobs=jobs,conditional_ablations=contrasts,
               completed_jobs=sum(j['completed'] for j in jobs),expected_jobs=len(jobs),performance_accepted=False)
    write(out/'report.json',value);return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--stage',choices=('screen','attribution'),default='screen')
    q.add_argument('--config',default='configs/b2.json');q.add_argument('--options',default='configs/r123_learning_v1.json');q.add_argument('--output',required=True)
    q=sub.add_parser('evaluate');q.add_argument('--root',required=True);q.add_argument('--validation',required=True)
    q.add_argument('--mode',choices=('rule','learned'),default='rule');q.add_argument('--policy')
    q.add_argument('--suite',choices=('development','test'),default='development');q.add_argument('--final-test',action='store_true');q.add_argument('--output',required=True)
    q=sub.add_parser('run');q.add_argument('--root',required=True);q.add_argument('--validation',required=True)
    q.add_argument('--reference',required=True);q.add_argument('--screen');q.add_argument('--arm',choices=ARMS,required=True);q.add_argument('--seed',required=True,type=int)
    q=sub.add_parser('report');q.add_argument('--root',required=True);q.add_argument('--output',required=True)
    for q in sub.choices.values():q.add_argument('--execute',action='store_true')
    a=p.parse_args(argv)
    if not a.execute:value=dict(executed=False,requested=vars(a))
    else:
        torch.set_num_threads(1);value={'prepare':prepare,'evaluate':evaluation,'run':run,'report':report}[a.command](a)
    print(json.dumps(value,indent=2,allow_nan=False));return value


if __name__=='__main__':main()
