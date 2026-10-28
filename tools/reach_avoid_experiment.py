"""Explicit Linux research: fresh latest-RL training with fixed development monitoring."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from project_paths import ROOT
from chapter3_bser.reach_avoid.learning_diagnostics import create_json,sha,settings,schedule


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--validation',required=True);p.add_argument('--output',required=True)
    p.add_argument('--episodes',type=int,default=500);p.add_argument('--train-cases',type=int,default=32)
    p.add_argument('--bootstrap-episodes',type=int,default=32);p.add_argument('--dev-cases',type=int,default=12)
    p.add_argument('--pretrain-updates',type=int,default=256)
    p.add_argument('--config',default=str(ROOT/'configs/b2.json'))
    p.add_argument('--options',default=str(ROOT/'configs/reach_avoid.json'))
    p.add_argument('--monitor-settings',default=str(ROOT/'configs/reach_avoid_monitor.json'))
    p.add_argument('--execute',action='store_true');args=p.parse_args(argv)
    if min(args.episodes,args.train_cases,args.bootstrap_episodes,args.dev_cases,args.pretrain_updates)<1:p.error('positive explicit budgets required')
    spec=settings(json.loads(Path(args.monitor_settings).read_text(encoding='utf-8')))
    plan=dict(executed=False,online_training_episodes=args.episodes,bootstrap_episodes=args.bootstrap_episodes,
        training_cases=args.train_cases,development_cases=args.dev_cases,evaluation_points=schedule(args.episodes,spec['eval_every']),
        other_algorithm_experiments=False,checkpoint_resume=False,final_test_used=False)
    if not args.execute:print(json.dumps(plan,indent=2));return plan
    from chapter3_bser.reach_avoid.workflow import admission
    from chapter3_bser.learning.scenarios import export
    source=admission(args.validation);out=Path(args.output).resolve()
    if out.exists():raise FileExistsError(out)
    if not out.is_relative_to(ROOT/'work/collision_terminal'):raise ValueError('fresh collision_terminal output required')
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    if config['max_steps']!=400:raise ValueError('declared experiment requires original 400-step episodes')
    out.mkdir(parents=True,exist_ok=False)
    plan.update(executed=True,source_identity=source,config_sha256=sha(args.config),options_sha256=sha(args.options),
        monitor_settings_sha256=sha(args.monitor_settings),train_seed=81001,development_seed=82001,
        bootstrap_max_steps=400*args.bootstrap_episodes,online_max_steps=400*args.episodes,
        development_max_steps=400*args.dev_cases*len(plan['evaluation_points']))
    create_json(out/'plan.json',plan)
    export(out/'training.json',split='train',count=args.train_cases,seed=81001,max_steps=400)
    export(out/'development.json',split='eval',count=args.dev_cases,seed=82001,max_steps=400)
    common=['--config',str(Path(args.config).resolve()),'--options',str(Path(args.options).resolve()),'--validation',str(Path(args.validation).resolve()),'--execute']
    jobs=[('collect',['--scenarios',str(out/'training.json'),'--episodes',str(args.bootstrap_episodes),'--output',str(out/'bootstrap'),'--behavior','mixed']),
          ('train',['--scenarios',str(out/'training.json'),'--episodes',str(args.episodes),'--output',str(out/'train'),
            '--dataset',str(out/'bootstrap'),'--pretrain-updates',str(args.pretrain_updates),'--dev-scenarios',str(out/'development.json'),
            '--monitor-settings',str(Path(args.monitor_settings).resolve())])]
    for name,tail in jobs:
        if any(sha(getattr(args,key))!=plan[key+'_sha256'] for key in ('config','options','monitor_settings')):
            raise ValueError('experiment inputs changed after plan freeze')
        command=[sys.executable,'-u','-m','chapter3_bser.reach_avoid.workflow',name,*common,*tail]
        create_json(out/(name+'_command.json'),dict(argv=command,source_identity=source))
        print('[experiment] starting '+name,flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
    create_json(out/'completed.json',dict(source_identity=source,online_training_episodes=args.episodes,
        result_sha256=sha(out/'train/completed.json'),report_sha256=sha(out/'train/monitoring/final_report/report.json'),
        performance_passed=False,safety_guarantee=False))
    print('[experiment] completed; report: '+str(out/'train/monitoring/final_report/report.html'),flush=True)
    return plan


if __name__=='__main__':main()
