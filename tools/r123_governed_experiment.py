"""Retest governed tracking on the same six scenes; previous result is read-only."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from project_paths import ROOT
from tools.path_option_experiment import read
from chapter3_bser.r123_repair.source import identity as parent_identity
from chapter3_bser.r123_governed.source import identity,admission,sha,digest,write,new_output
from chapter3_bser.r123_governed.workflow import evaluate,metric
from chapter3_bser.path_option.workflow import task_config,check_splits
from chapter3_bser.r123.options import options


def prepare(a):
    old=Path(a.previous);src=identity();out=new_output(a.output)
    config=task_config(read(old/'plan/config.json'));opts=options(read(old/'plan/options.json'))
    scenes=read(old/'plan/development.json');base=old/'no_recovery';done=read(base/'completed.json')
    if done['source']!=parent_identity() or done['variant']!='no_recovery' or done['seed']!=982001:
        raise ValueError('matching previous no_recovery result required')
    if done['config_hash']!=digest(config) or done['options_hash']!=digest(opts):raise ValueError('baseline settings mismatch')
    ids=[s['scenario_id'] for s in scenes['scenarios']]
    if len(ids)!=6 or done['scenario_ids']!=ids:raise ValueError('same six scenes required')
    if any(s['max_steps']!=config['max_steps'] for s in scenes['scenarios']):raise ValueError('scene time budget differs')
    check_splits([],scenes['scenarios'],[])
    for name,key in [('episodes.jsonl','episodes_sha256'),('metrics.jsonl','metrics_sha256'),('summary.json','summary_sha256')]:
        if sha(base/name)!=done[key]:raise ValueError('historical artifact changed: '+name)
    episodes=[json.loads(x) for x in (base/'episodes.jsonl').read_text(encoding='utf-8').splitlines()]
    if [r['scenario_id'] for r in episodes]!=ids:raise ValueError('historical episode order mismatch')
    out.mkdir(parents=True)
    write(out/'config.json',config);write(out/'options.json',opts);write(out/'development.json',scenes)
    write(out/'previous.json',dict(source=done['source'],rows=[metric(r,config['max_steps']) for r in episodes],
        episodes_sha256=done['episodes_sha256'],diagnostic_coverage='previous traces do not contain candidate check diagnostics'))
    result=dict(source=src,episodes=6,step_cap=6*config['max_steps'],seed=982001,training_allowed=False,
        inputs={p.name:sha(p) for p in out.glob('*.json')})
    write(out/'plan.json',result);return result


def frozen(path):
    p=Path(path);m=read(p/'plan.json')
    if m['source']!=identity():raise ValueError('new governed plan required')
    for name,h in m['inputs'].items():
        if sha(p/name)!=h:raise ValueError('frozen input changed')
    return p


def evaluation(a):
    admission(a.validation);p=frozen(a.root)
    return evaluate(read(p/'config.json'),read(p/'development.json')['scenarios'],read(p/'options.json'),'governed',new_output(a.output))


def report(a):
    p=frozen(a.root);r=Path(a.result);done=read(r/'completed.json')
    if done['source']!=identity() or done['variant']!='governed' or done['seed']!=982001:raise ValueError('wrong result identity')
    if done['config_hash']!=digest(read(p/'config.json')) or done['options_hash']!=digest(read(p/'options.json')):raise ValueError('settings mismatch')
    for name,key in [('episodes.jsonl','episodes_sha256'),('metrics.jsonl','metrics_sha256'),('summary.json','summary_sha256')]:
        if sha(r/name)!=done[key]:raise ValueError('result changed')
    old=read(p/'previous.json')['rows'];new=[json.loads(x) for x in (r/'metrics.jsonl').read_text(encoding='utf-8').splitlines()]
    expected=[s['scenario_id'] for s in read(p/'development.json')['scenarios']]
    if [x['scenario_id'] for x in old]!=expected or [x['scenario_id'] for x in new]!=expected or done['scenario_ids']!=expected:raise ValueError('scenes unpaired')
    delta={k:dict(per_scene=[b[k]-a[k] for a,b in zip(old,new)],mean=float(np.mean([b[k]-a[k] for a,b in zip(old,new)]))) for k in
        ('success','found','collision','timeout','steps','search_wait_steps','search_stationary_steps','search_slow_distant_steps','search_brake_steps')}
    out=new_output(a.output);out.mkdir(parents=True)
    value=dict(source=identity(),previous=old,governed=new,paired_differences=delta,training_allowed=False,performance_accepted=False)
    write(out/'report.json',value);return value


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--previous',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('evaluate');q.add_argument('--root',required=True);q.add_argument('--validation',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('report');q.add_argument('--root',required=True);q.add_argument('--result',required=True);q.add_argument('--output',required=True)
    for q in sub.choices.values():q.add_argument('--execute',action='store_true')
    a=p.parse_args(argv)
    if not a.execute:v=dict(executed=False,requested=vars(a))
    else:
        torch.set_num_threads(1);v=dict(prepare=prepare,evaluate=evaluation,report=report)[a.command](a)
    print(json.dumps(v,indent=2,allow_nan=False));return v


if __name__=='__main__':main()
