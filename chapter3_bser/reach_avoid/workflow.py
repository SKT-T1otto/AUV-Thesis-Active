"""Manual collect/train/evaluate for the independent reach-avoid variant."""
import argparse,copy,hashlib,json
from pathlib import Path
import numpy as np
import torch
from project_paths import ROOT
from chapter3_bser.learning.provenance import verify_reviewed_changes,require_review,research_readiness
from chapter3_bser.learning.map_suite import require_linux_validation
from chapter3_bser.learning.config import scenario_fingerprint
from chapter3_bser.learning.checkpoint import state_digest
from chapter3_bser.learning.reference import ZeroResidualPolicy
from chapter3_bser.safe_rl.workflow import inputs
from chapter3_bser.reach_avoid.options import SCHEMA,validate
from chapter3_bser.reach_avoid.learner import ReachAvoidLearner
from chapter3_bser.reach_avoid.runtime import ReachAvoidRuntime
from chapter3_bser.reach_avoid import dataset


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,data):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2,allow_nan=False)


def admission(validation):
    source=verify_reviewed_changes();research_readiness(source)
    if source['profile'] not in ('active_reach_avoid_v1','active_reach_avoid_geometry_v2'):raise ValueError('current independent reach-avoid source required')
    path=Path(validation);path=path/'validation.json' if path.is_dir() else path
    require_linux_validation(path,source)
    return source


def execute(args):
    source=admission(args.validation);options=validate(read(args.options) if args.options else None)
    training=args.command=='train';collecting=args.command=='collect'
    config,scenes=inputs(args.config,args.scenarios,training=training or collecting,episodes=args.episodes,cases=args.cases)
    output=Path(args.output).resolve()
    if not output.is_relative_to(ROOT/'work/collision_terminal'):raise ValueError('fresh collision_terminal output required')
    if output.exists():raise FileExistsError(output)
    if args.intervention!='none' and args.command!='evaluate':raise ValueError('interventions are evaluation-only')
    if training and args.mode!='learned':raise ValueError('training must select the new learned reach-avoid variant')
    if collecting and args.policy:raise ValueError('collection does not restore learned checkpoints')
    torch.set_num_threads(1)
    learner=None
    training_ids=[];fingerprints=[];pretrain=[];dataset_inputs=[]
    if training:
        if args.policy:raise ValueError('training checkpoint resume is unsupported')
        if not args.dataset:raise ValueError('independent train-split factual --dataset required')
        learner=ReachAvoidLearner(options,ablation=args.ablation)
        for folder in args.dataset:
            episodes,old_scenes=dataset.load_collection(folder,source,options)
            for episode in episodes:learner.add_episode(episode)
            training_ids.extend(s['scenario_id'] for s in old_scenes);fingerprints.extend(scenario_fingerprint(s) for s in old_scenes)
            dataset_inputs.append(dict(path=str(Path(folder).resolve()),receipt_sha256=sha(Path(folder)/'dataset.json')))
        coverage=learner.replay.counts
        if len(learner.replay)<options['warmup_samples'] or coverage['reach']<1 or coverage['collision']<1:
            raise ValueError('retained replay needs factual collision and successful goal examples; safe-only or evicted evidence is insufficient')
        if args.pretrain_updates<options['minimum_updates']:raise ValueError('explicit pretraining budget below readiness minimum')
    elif args.mode=='learned':
        if not args.policy:raise ValueError('learned evaluation requires this algorithm new policy file')
        metadata=read(str(args.policy)+'.json')
        if (metadata.get('schema')!=SCHEMA or metadata.get('source_identity')!=source
            or metadata.get('sha256')!=sha(args.policy) or metadata.get('resume_supported') is not False):raise ValueError('policy identity/hash mismatch before tensor load')
        if metadata.get('ablation')!=args.ablation:raise ValueError('trained ablation differs; use an explicit evaluation intervention')
        training_ids=metadata['training_scenario_ids'];fingerprints=metadata['training_fingerprints']
        if set(training_ids)&{s['scenario_id'] for s in scenes} or set(fingerprints)&{scenario_fingerprint(s) for s in scenes}:
            raise ValueError('evaluation overlaps reach-avoid training')
        learner=ReachAvoidLearner(options,ablation=args.ablation)
        state=torch.load(args.policy,map_location='cpu',weights_only=True)
        if state_digest(state)!=metadata['state_sha256']:raise ValueError('policy state changed')
        learner.load_for_evaluation(state);learner.intervention=args.intervention
    elif args.policy or args.intervention!='none':raise ValueError('rule evaluation does not load or alter policies')
    output.mkdir(parents=True,exist_ok=False)
    run=dict(schema=SCHEMA,source_identity=source,command=args.command,options=options,config=config,
        mode=args.mode,ablation=args.ablation,evaluation_intervention=args.intervention,behavior=args.behavior,
        scenario_manifest_sha256=sha(args.scenarios),scenario_ids=[s['scenario_id'] for s in scenes],
        task_policy='D2_ZERO',task_network_training=False,independent_reach_avoid_training=training,
        checkpoint_resume=False,policy_sha256=sha(args.policy) if args.policy else None,dataset_inputs=dataset_inputs,
        original_reward_dynamics_protocol_unchanged=True,performance_passed=False,safety_guarantee=False)
    write(output/'run.json',run);write(output/'scenarios.json',dict(scenarios=scenes))
    if training:
        for update in range(args.pretrain_updates):
            metric=learner.update();pretrain.append(metric)
            if (update+1)%32==0:print(f'[reach-avoid pretrain] {update+1}/{args.pretrain_updates}',flush=True)
    records=[];files=[];metrics=[];steps=0
    with (output/'episodes.jsonl').open('x',encoding='utf-8') as stream:
        for episode in range(args.episodes):
            require_review(source);scene=copy.deepcopy(scenes[episode%len(scenes)])
            behavior=('nominal' if episode%2==0 else 'explore') if collecting and args.behavior=='mixed' else args.behavior if collecting else 'learned' if learner else 'rule'
            runtime=ReachAvoidRuntime(config,scene,seed=int(scene['scenario_seed']),episode_id=episode+1,
                options=options,learner=learner,behavior=behavior,ablation='no_goal_repair' if args.intervention=='no_goal_repair' else args.ablation)
            print(f'[reach-avoid] episode={episode+1}/{args.episodes} scene={scene["scenario_id"]} behavior={behavior}',flush=True)
            try:
                while not runtime.terminal:
                    runtime.advance(ZeroResidualPolicy(),explore=training);steps+=1
                    if runtime.step%25==0:print(f'[reach-avoid] episode={episode+1} step={runtime.step}/{config["max_steps"]}',flush=True)
                summary=runtime.summary()
                if collecting:files.append(dataset.save(output/f'episode_{episode+1:06d}.npz',runtime.ra_records))
                if training:
                    learner.add_episode(runtime.ra_records)
                    for _ in range(max(1,len(runtime.ra_records)//options['update_frequency'])):
                        metric=learner.update()
                        if metric is not None:metrics.append(metric)
                    training_ids.append(scene['scenario_id']);fingerprints.append(scenario_fingerprint(scene))
            finally:runtime.close()
            require_review(source);summary.update(episode=episode+1,reach_avoid_training=training)
            stream.write(json.dumps(summary,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();records.append(summary)
            print(f'[reach-avoid] finished={episode+1} termination={summary["termination_reason"]} success={summary["success"]}',flush=True)
    if training:
        write(output/'learning_metrics.json',dict(pretrain=pretrain,online=metrics,updates=learner.updates,retained_strata=learner.replay.counts))
        if learner.ready:
            state=learner.state();path=output/'reach_avoid_policy.pt';torch.save(state,path)
            write(str(path)+'.json',dict(schema=SCHEMA,source_identity=source,sha256=sha(path),state_sha256=state_digest(state),
                ablation=args.ablation,training_scenario_ids=sorted(set(training_ids)),training_fingerprints=sorted(set(fingerprints)),
                resume_supported=False,old_checkpoint_relabelled=False,performance_passed=False))
    if collecting:write(output/'dataset.json',dict(schema=SCHEMA,source_identity=source,complete=True,files=files,
        run_sha256=sha(output/'run.json'),scenarios_sha256=sha(output/'scenarios.json')))
    result=dict(schema=SCHEMA,source_identity=source,completed=True,episodes=len(records),physical_steps=steps,
        collisions=sum(r['termination_reason']=='obstacle_collision' for r in records),successes=sum(bool(r['success']) for r in records),
        reach_avoid_updates=learner.updates if learner else 0,task_network_updates=0,performance_passed=False,safety_guarantee=False)
    require_review(source);write(output/'completed.json',result);return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);subs=parser.add_subparsers(dest='command',required=True)
    for command in ('collect','train','evaluate'):
        p=subs.add_parser(command)
        for name in ('config','scenarios','output','validation'):p.add_argument('--'+name,required=True)
        p.add_argument('--episodes',type=int,required=True);p.add_argument('--cases',nargs='+');p.add_argument('--options')
        p.add_argument('--mode',choices=('rule','learned'),default='learned' if command=='train' else 'rule')
        p.add_argument('--behavior',choices=('nominal','explore','mixed'),default='mixed')
        p.add_argument('--ablation',choices=('full','safety_only','no_goal_repair'),default='full')
        p.add_argument('--intervention',choices=('none','no_goal_repair','zero_residual'),default='none')
        p.add_argument('--policy');p.add_argument('--dataset',nargs='+');p.add_argument('--pretrain-updates',type=int,default=256)
        p.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    if args.episodes<1 or args.pretrain_updates<1:parser.error('positive explicit run/update budgets required')
    result=execute(args) if args.execute else dict(executed=False,algorithm=SCHEMA,**vars(args))
    print(json.dumps(result,ensure_ascii=False,indent=2));return result

if __name__=='__main__':main()
