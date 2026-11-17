"""Paired tracking repairs: import baseline; run only three new rule arms."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from project_paths import ROOT
from tools.path_option_experiment import read
from chapter3_bser.r123_governed.source import identity as parent_identity
from chapter3_bser.r123_tracking.source import identity,admission,sha,digest,write,new_output
from chapter3_bser.r123_tracking.runtime import ARMS
from chapter3_bser.r123_tracking.workflow import evaluate,metric
from chapter3_bser.path_option.workflow import task_config,check_splits
from chapter3_bser.r123.options import options

NEW_ARMS=('path_only','feedforward_only','both')
PAIRED=('success','found','collision','timeout','steps','search_stationary_steps',
        'search_slow_distant_steps','search_brake_steps','capture_contact_steps','handoff')


def verify_result(base,done):
    for name in ('episodes','metrics','summary'):
        suffix='.json' if name=='summary' else '.jsonl'
        if sha(base/(name+suffix))!=done[name+'_sha256']:raise ValueError('result changed: '+name)


def prepare(a):
    old=Path(a.previous);out=new_output(a.output);src=identity()
    config=task_config(read(old/'plan/config.json'));opts=options(read(old/'plan/options.json'))
    scenes=read(old/'plan/development.json');done=read(old/'governed/completed.json')
    oldplan=read(old/'plan/plan.json')
    if oldplan['source']!=parent_identity():raise ValueError('governed parent plan required')
    for name,h in oldplan['inputs'].items():
        if sha(old/'plan'/name)!=h:raise ValueError('parent plan input changed')
    if done['source']!=parent_identity() or done['variant']!='governed' or done['seed']!=982001:raise ValueError('verified governed baseline required')
    if done['config_hash']!=digest(config) or done['options_hash']!=digest(opts):raise ValueError('baseline settings mismatch')
    ids=[s['scenario_id'] for s in scenes['scenarios']]
    if len(ids)!=6 or done['scenario_ids']!=ids or config['max_steps']!=400:raise ValueError('same six 400-step scenes required')
    if any(s['max_steps']!=400 for s in scenes['scenarios']):raise ValueError('scene step budget mismatch')
    check_splits([],scenes['scenarios'],[]);verify_result(old/'governed',done)
    episodes=[json.loads(x) for x in (old/'governed/episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    if [r['scenario_id'] for r in episodes]!=ids or any(r['task_network_trained'] for r in episodes):raise ValueError('baseline episode mismatch')
    out.mkdir(parents=True)
    write(out/'config.json',config);write(out/'options.json',opts);write(out/'development.json',scenes)
    write(out/'baseline.json',dict(source=done['source'],rows=[metric(r,400) for r in episodes],episodes_sha256=done['episodes_sha256']))
    value=dict(source=src,seed=982001,episodes_per_arm=6,new_arms=list(NEW_ARMS),
        new_physical_step_cap=7200,baseline_reused=True,training_allowed=False,
        inputs={p.name:sha(p) for p in out.glob('*.json')})
    write(out/'plan.json',value);return value


def frozen(path):
    p=Path(path);m=read(p/'plan.json')
    if m['source']!=identity():raise ValueError('current tracking plan required')
    if m['new_arms']!=list(NEW_ARMS) or m['seed']!=982001:raise ValueError('wrong experiment plan')
    for name,h in m['inputs'].items():
        if sha(p/name)!=h:raise ValueError('frozen input changed')
    return p


def evaluation(a):
    admission(a.validation);p=frozen(a.root)
    return evaluate(read(p/'config.json'),read(p/'development.json')['scenarios'],
                    read(p/'options.json'),a.arm,new_output(a.output))


def status(a):
    p=frozen(a.root);base=Path(a.results)
    return dict(baseline_reused=True,jobs=[dict(arm=arm,completed=(base/arm/'completed.json').is_file(),
        recorded_episodes=len((base/arm/'metrics.jsonl').read_text(encoding='utf-8').splitlines())
        if (base/arm/'metrics.jsonl').exists() else 0) for arm in NEW_ARMS],read_only=True)


def report(a):
    p=frozen(a.root);rows={'baseline':read(p/'baseline.json')['rows']}
    ids=[s['scenario_id'] for s in read(p/'development.json')['scenarios']]
    if [r['scenario_id'] for r in rows['baseline']]!=ids:raise ValueError('baseline not paired')
    for arm in NEW_ARMS:
        base=Path(a.results)/arm;done=read(base/'completed.json')
        if done['source']!=identity() or done['variant']!=arm or done['seed']!=982001:raise ValueError('wrong arm identity')
        if done['switches']!=dict(zip(('path','feedforward'),ARMS[arm])):raise ValueError('wrong switches')
        if done['config_hash']!=digest(read(p/'config.json')) or done['options_hash']!=digest(read(p/'options.json')):raise ValueError('different settings')
        verify_result(base,done)
        rows[arm]=[json.loads(x) for x in (base/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
        if [r['scenario_id'] for r in rows[arm]]!=ids or done['scenario_ids']!=ids:raise ValueError('scenes not paired')
    deltas={arm:{k:dict(per_scene=[n[k]-b[k] for b,n in zip(rows['baseline'],rows[arm])],
                      mean=float(np.mean([n[k]-b[k] for b,n in zip(rows['baseline'],rows[arm])])))
                 for k in PAIRED} for arm in NEW_ARMS}
    interaction={k:[b[k]-p0[k]-f[k]+o[k] for o,p0,f,b in zip(rows['baseline'],rows['path_only'],rows['feedforward_only'],rows['both'])] for k in PAIRED}
    out=new_output(a.output);out.mkdir(parents=True)
    value=dict(source=identity(),complete_jobs=3,expected_new_jobs=3,baseline_reused=True,
        arms=rows,paired_differences=deltas,factorial_interaction_per_scene=interaction,
        training_allowed=False,performance_accepted=False)
    write(out/'report.json',value);return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--previous',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('evaluate');q.add_argument('--root',required=True);q.add_argument('--validation',required=True);q.add_argument('--output',required=True);q.add_argument('--arm',choices=NEW_ARMS,required=True)
    q=sub.add_parser('report');q.add_argument('--root',required=True);q.add_argument('--results',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('status');q.add_argument('--root',required=True);q.add_argument('--results',required=True)
    for q in sub.choices.values():q.add_argument('--execute',action='store_true')
    a=p.parse_args(argv)
    if not a.execute:value=dict(executed=False,requested=vars(a))
    else:
        torch.set_num_threads(1);value=dict(prepare=prepare,evaluate=evaluation,report=report,status=status)[a.command](a)
    print(json.dumps(value,indent=2,allow_nan=False));return value


if __name__=='__main__':main()
